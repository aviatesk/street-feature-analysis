# %%

import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.python.framework import ops

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DIR_PATH = os.path.join('..')
DEFAULT_MODEL_FILE_PATH = os.path.join(DIR_PATH, 'models',
                                       '512-avg-0.5vs1-15.best.default.h5')

H, W, C = 224, 224, 3
CITIES = ('london', 'moscow', 'nyc', 'paris', 'vancouver', 'beijing', 'kyoto',
          'seoul', 'singapore', 'tokyo')
LABEL2CLASS1 = [(0, 'western'), (1, 'eastern')]
LABEL2CLASS2 = [(i, city_name) for i, city_name in enumerate(CITIES)]


def load_image(image_path, target_size=(H, W), preprocessing=False):
    '''
    load and preprocess an image from its path
    '''
    x = image.load_img(image_path, target_size=target_size)

    if preprocessing:

        x = image.img_to_array(x)[np.newaxis]
        gen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            rescale=1. / 255,
        ).flow(
            x, batch_size=1, shuffle=False)
        x = next(gen)

    return x


def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def l2_normalize(x):
    '''
    normalize a tensor by its L2 norm
    '''
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def set_guided_model(model_file_path):
    '''
    return a model, whose gradient function for all the ReLU activations according to guided backpropagation are changed
    '''

    if "GuidedBackProp" not in ops._gradient_registry._registry:

        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * tf.cast(
                op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = load_model(model_file_path)

    return new_model


def guided_backpropagation(guided_model, preprocessed_input, layer_name):
    '''
    Guided Backpropagation method for visualizing saliency of input image
    '''
    layer_output = guided_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, guided_model.input)[0]

    backprop_function = K.function(
        [guided_model.input, K.learning_phase()], [grads])
    grads_val = backprop_function([preprocessed_input, 0])[0]

    return grads_val


def compute_grad_cam(model,
                     preprocessed_input,
                     out,
                     cls,
                     layer_name,
                     output_shape=(H, W),
                     normalize=False):
    '''
    ### Grad-CAM method for visualizing saliency of input image

    ### argments:
    - `out`: the kind or outputs (`1`: output1, `2`: output2)
    - `output_shape`: tuple of output image shape in (h, w) order
    '''
    y_c = model.output[out][0, cls]
    target_conv_output = model.get_layer(layer_name).output
    grads = K.gradients(y_c, target_conv_output)[0]

    if normalize:  # normalize if necessary
        grads = l2_normalize(grads)

    gradient_function = K.function([model.input], [target_conv_output, grads])
    output, grads_val = gradient_function([preprocessed_input])

    weights = np.mean(grads_val[0, :, :, :], axis=(0, 1))
    cam = np.dot(output[0, :], weights)

    # process CAM
    cam = cv2.resize(cam, output_shape[::-1], cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam


def compute_grad_cam_batch(input_model, images, classes, layer_name):
    '''
    Grad-CAM method for visualizing input saliency, same as `compute_grad_cam` but process multiple images in one run
    '''

    loss = tf.gather_nd(input_model.output,
                        np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function(
        [input_model.input, K.learning_phase()], [layer_output, grads])

    target_conv_output, grads_val = gradient_fn([images, 0])
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', target_conv_output, weights)

    # Process CAMs
    new_cams = np.empty((images.shape[0], H, W))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (H, W), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    return new_cams


# %%

model = load_model(DEFAULT_MODEL_FILE_PATH)
guided_model = set_guided_model(DEFAULT_MODEL_FILE_PATH)

localized_cls1, localized_cls2 = 0, 0
top_n = 3
layer_name = 'block5_conv3'
alpha = 0.4

dir_path = os.path.join(DIR_PATH, 'data', 'splitted', 'test')
output_dir = os.path.join(DIR_PATH, 'assets')

# %%

cities = ('london', 'moscow', 'nyc', 'paris', 'vancouver')
target_numss = (
    [0, 14, 10, 40, 46],
    [4, 26, 30, 13, 27],
    [1, 5, 27, 34, 50],
    [2, 13, 24, 30, 41],
    [0, 8, 34, 16, 28],
)

plt.figure(figsize=(18, 40), dpi=500)

for j, (city_name, target_nums) in tqdm(enumerate(zip(cities, target_numss))):

    for jj, target_num in enumerate(target_nums):
        image_path = os.path.join(dir_path, city_name,
                                  '{}.png'.format(target_num))
        # load and preprocess the input image
        preprocessed_input = load_image(
            image_path, target_size=(H, W), preprocessing=True)

        # make prediction
        y1_pred, y2_pred = model.predict(preprocessed_input)
        sorted_indices1 = np.argsort(y1_pred[0])[::-1]
        sorted_indices2 = np.argsort(y2_pred[0])[::-1]
        cls1 = sorted_indices1[localized_cls1]
        cls2 = sorted_indices2[localized_cls2]
        cls1_name = LABEL2CLASS1[cls1][1].capitalize()
        cls2_name = LABEL2CLASS2[cls2][1].capitalize(
        ) if LABEL2CLASS2[cls2][1] != 'nyc' else 'NYC'

        p = y2_pred[0][sorted_indices2[0]] * 100
        original_image = load_image(
            image_path, target_size=None, preprocessing=False)
        output_shape = original_image.size[::-1]

        gc2 = compute_grad_cam(
            model,
            preprocessed_input,
            1,
            cls2,
            layer_name,
            output_shape=output_shape)

        # set figure
        plt.subplot(10, 5, 10 * j + jj + 1)
        city_name = city_name.capitalize() if city_name != 'nyc' else 'NYC'
        plt.title(city_name)
        plt.axis('off')
        plt.imshow(original_image)
        plt.subplot(10, 5, 10 * j + jj + 6)
        plt.title('\'{}\': {:.2f}'.format(cls2_name, p))
        plt.axis('off')
        plt.imshow(original_image)
        plt.imshow(gc2, cmap='jet', alpha=alpha)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, 'western.png'), dpi=50, bbox_inches='tight')

# %%

cities = ('beijing', 'kyoto', 'seoul', 'singapore', 'tokyo')
target_numss = ([9, 13, 25, 35, 23], [4, 18, 38, 49, 25], [0, 5, 16, 26, 30],
                [1, 3, 11, 20, 25], [32, 6, 7, 18, 28])

plt.figure(figsize=(18, 40), dpi=500)

for j, (city_name, target_nums) in tqdm(enumerate(zip(cities, target_numss))):

    for jj, target_num in enumerate(target_nums):
        image_path = os.path.join(dir_path, city_name,
                                  '{}.png'.format(target_num))
        # load and preprocess the input image
        preprocessed_input = load_image(
            image_path, target_size=(H, W), preprocessing=True)

        # make prediction
        y1_pred, y2_pred = model.predict(preprocessed_input)
        sorted_indices1 = np.argsort(y1_pred[0])[::-1]
        sorted_indices2 = np.argsort(y2_pred[0])[::-1]
        cls1 = sorted_indices1[localized_cls1]
        cls2 = sorted_indices2[localized_cls2]
        cls1_name = LABEL2CLASS1[cls1][1].capitalize()
        cls2_name = LABEL2CLASS2[cls2][1].capitalize(
        ) if LABEL2CLASS2[cls2][1] != 'nyc' else 'NYC'

        p = y2_pred[0][sorted_indices2[0]] * 100
        original_image = load_image(
            image_path, target_size=None, preprocessing=False)
        output_shape = original_image.size[::-1]

        gc2 = compute_grad_cam(
            model,
            preprocessed_input,
            1,
            cls2,
            layer_name,
            output_shape=output_shape)

        # set figure
        plt.subplot(10, 5, 10 * j + jj + 1)
        city_name = city_name.capitalize() if city_name != 'nyc' else 'NYC'
        plt.title(city_name)
        plt.axis('off')
        plt.imshow(original_image)
        plt.subplot(10, 5, 10 * j + jj + 6)
        plt.title('\'{}\': {:.2f}'.format(cls2_name, p))
        plt.axis('off')
        plt.imshow(original_image)
        plt.imshow(gc2, cmap='jet', alpha=alpha)

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, 'eastern.png'), dpi=50, bbox_inches='tight')

# %%
