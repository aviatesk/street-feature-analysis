from places365_vgg16_keras.vgg16_places_365 import VGG16_Places365
from places365_vgg16_keras.vgg16_hybrid_places_1365 import VGG16_Hybrid_1365
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.layers import BatchNormalization, Activation, Flatten, Dropout, Dense
from keras.optimizers import SGD

H, W, C = 224, 224, 3
'../models/fine-tuned/512-avg-0.5vs1-15.best.h5'

def _get_offtop(weights='places', input_shape=(H, W, C), pooling=None):

    if weights == 'places':
        return VGG16_Places365(
            weights='places',
            include_top=False,
            input_shape=input_shape,
            pooling=pooling,
        )

    elif weights == 'hybrid':
        return VGG16_Hybrid_1365(
            weights='places',
            include_top=False,
            input_shape=input_shape,
            pooling=pooling,
        )

    elif weights is None or weights == 'imagenet':
        return VGG16(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
            pooling=pooling)

    else:
        raise ValueError(
            '`weights` should be `places` or `hybrid` or `imagent`, or None')


def create_model(
        n_class2,
        n_class1=2,
        weights='places',
        input_shape=(H, W, C),
        pooling='avg',
        units=512,
        drop_rate=0.5,
        optimizer='nadam',
        loss_weights=(1, 1),
):
    '''
    creates VGG16 architecture network, for the multi-output street classification

    augments:
    weights: None or str ('places', 'hybrid', 'imagenet'), if None the random initialized weights will be returned
    - pooling: the pooling after all the convolutions (None, 'avg', 'max')
    - units: the number of units at the layer after the final convolutional layer (let's go 512 with pooling 'avg' or 4096 without pooling)
    - optimizer: 'nadam', 'rmsprop', etc...
    - loss_weights: loss weights for between output1 and output2
    '''

    bottleneck_model = _get_offtop(
        weights=weights, input_shape=input_shape, pooling=pooling)
    model_input = bottleneck_model.input

    x = bottleneck_model.output
    if not pooling:
        x = Flatten()(x)
        x = Dense(units)(x)  # Dense(units, use_bias=False)(x)
    else:
        x = Dense(units)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop_rate)(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop_rate)(x)

    output1 = Dense(n_class1, activation='softmax', name='output1')(x)
    output2 = Dense(n_class2, activation='softmax', name='output2')(x)

    model = Model(model_input, [output1, output2])

    if weights:
        for layer in bottleneck_model.layers:
            layer.trainable = False

    loss_weight1, loss_weight2 = loss_weights
    model.compile(
        optimizer=optimizer,
        loss={
            'output1': 'categorical_crossentropy',
            'output2': 'categorical_crossentropy',
        },
        loss_weights={
            'output1': loss_weight1,
            'output2': loss_weight2
        },
        metrics=['acc'],
    )

    return model


def set_fine_tune(
        model_file_path,
        freeze=11,
        optimizer=SGD(lr=1e-4, momentum=0.9, nesterov=True),
        loss_weights=None,
):
    '''
    load a saved model and set it up for fine-tuning

    argments:
    - freeze: the number of layers, which will be freezed during training (11 or 15)
    - loss_weights: if specified, set the new loss_weights, otherwise use the same loss_weights as the loaded model
    '''

    model = load_model(model_file_path, compile=True)

    for layer in model.layers[:freeze]:
        layer.trainable = False
    for layer in model.layers[freeze:]:
        layer.trainable = True

    # compile again for fine-tuning
    model.compile(
        optimizer=optimizer,
        loss=model.loss,
        loss_weights=loss_weights if loss_weights else model.loss_weights,
        metrics=['acc'],
    )

    return model


if __name__ == '__main__':
    pass
