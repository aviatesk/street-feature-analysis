# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

H, W, C = 224, 224, 3
WESTERNS = ('london', 'moscow', 'nyc', 'paris', 'vancouver')
EASTERNS = ('beijing', 'kyoto', 'seoul', 'singapore', 'tokyo')


def create_generator(X, ys, mode='train', batch_size=32, random_state=42):
    '''
    create Keras ImageDataGenerator with multiple outputs

    argments
    - mode: specifies training or validate or test, given as str ('train', 'validate', 'test', 'predict') or integer (0: 'train', 1: 'validate', 2: 'test', 3: 'predict')

    returns: generator, steps
    - generator: generator, which takes a single input image and multiple outputs and then generates a batch of augmented images and multiple labels
    - steps: the default steps of the generator (same as len(generator) of an usual Keras NumpyArrayIterator object)
    '''

    mode = 'train' if mode == 0 else\
           'validate' if mode == 1 else\
           'test' if mode == 2 else\
           'predict' if mode == 3 else\
           mode

    if mode == 'train':

        gen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
        )

    elif mode == 'validate' or mode == 'predict':
        gen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
        )

    else:
        raise ValueError(
            '`mode` ill defined or mode `test` will not work, use the opened X_test'
        )

    if mode == 'train' or mode == 'validate':

        def generator(X, ys):
            gens = [
                gen.flow(X, y, batch_size=batch_size, seed=random_state)
                for y in ys
            ]
            while True:
                X_ys_gen = [gen.next() for gen in gens]
                X_gen = X_ys_gen[0][0]
                ys_gen = [y for X, y in X_ys_gen]
                yield X_gen, ys_gen

    elif mode == 'predict':

        def generator(X, ys):
            gens = [
                gen.flow(
                    X,
                    y,
                    batch_size=batch_size,
                    shuffle=False,
                    seed=random_state) for y in ys
            ]
            while True:
                X_ys_gen = [gen.next() for gen in gens]
                X_gen = X_ys_gen[0][0]
                yield X_gen

    default_steps = X.shape[0] // batch_size + 1

    return generator(X, ys), default_steps


def create_generator_gc(
        gc_path='.',
        cities=('london', 'moscow', 'nyc', 'paris', 'vancouver', 'beijing',
                'kyoto', 'seoul', 'singapore', 'tokyo'),
        mode='train',
        batch_size=32,
        random_state=42,
):
    '''
    create Keras ImageDataGenerator with multiple outputs, which will avoid using up memory, using ImageDataGenerator.flow_from_directory method

    argments
    - mode: specifies training or validate or test, given as str ('train', 'validate', 'test', 'predict') or integer (0: 'train', 1: 'validate', 2: 'test', 3: 'predict')

    returns: generator, steps, (label2class1, label2class2)
    - generator: generator, which takes a single input image and multiple outputs and then generates a batch of augmented images and multiple labels
    - steps: the default steps of the generator (same as len(generator) of an usual Keras NumpyArrayIterator object)
    - label2class1: label information about Western or Eastern
    - label2class2: label information about city names
    '''

    mode = 'train' if mode == 0 else\
           'validate' if mode == 1 else\
           'test' if mode == 2 else\
           'predict' if mode == 3 else\
           mode

    dirc_path = os.path.join(gc_path, 'data', mode) if mode != 'predict' else\
                os.path.join(gc_path, 'data', 'test')

    label2class1 = [(0, 'western'), (1, 'eastern')]
    label2class2 = [(i, city_name) for i, city_name in enumerate(cities)]

    if mode == 'train':

        gen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            rescale=1. / 255,
        )

    elif mode == 'validate' or mode == 'predict':
        gen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            rescale=1. / 255,
        )

    elif mode == 'test':
        gen = ImageDataGenerator(rescale=1. / 255, )

    else:
        raise ValueError('argment `mode` ill specified')

    if mode == 'train' or mode == 'validate':

        def generator(gen):
            generator = gen.flow_from_directory(
                dirc_path,
                target_size=(H, W),
                color_mode='rgb',
                classes=cities,
                class_mode='categorical',
                shuffle=True,
                batch_size=batch_size,
                seed=random_state,
            )

            while True:
                X, y2 = generator.next()
                y1 = to_categorical(
                    np.array(
                        list(
                            map(lambda x: 0 if cities[x] in WESTERNS else 1,
                                np.argmax(y2, axis=1)))),
                    num_classes=2,
                )
                ys = [y1, y2]
                yield X, ys

    elif mode == 'test':

        def generator(gen):
            generator = gen.flow_from_directory(
                dirc_path,
                target_size=(H, W),
                color_mode='rgb',
                classes=cities,
                class_mode='categorical',
                shuffle=False,
                batch_size=batch_size,
                seed=random_state,
            )

            while True:
                X, y2 = generator.next()
                y1 = to_categorical(
                    np.array(
                        list(
                            map(lambda x: 0 if cities[x] in WESTERNS else 1,
                                np.argmax(y2, axis=1)))),
                    num_classes=2,
                )
                ys = [y1, y2]
                yield X, ys

    elif mode == 'predict':

        def generator(gen):
            generator = gen.flow_from_directory(
                dirc_path,
                target_size=(H, W),
                color_mode='rgb',
                classes=cities,
                class_mode='categorical',
                shuffle=False,
                batch_size=batch_size,
                seed=random_state,
            )

            while True:
                X, y2 = generator.next()
                yield X

    num = len(glob.glob(os.path.join(dirc_path, '*', '*')))
    default_steps = num // batch_size + 1

    return generator(gen), default_steps, (label2class1, label2class2)


def get_test_data(
        gc_path='.',
        cities=('london', 'moscow', 'nyc', 'paris', 'vancouver', 'beijing',
                'kyoto', 'seoul', 'singapore', 'tokyo'),
        input_shape=(H, W, C),
        batch_size=32,
        random_state=42,
):
    '''
    obtain test labels, assuming to be used with create_generator_gc(mode='predict') called with keras.models.Model.predict_generator

    return: (y1_test, y2_test), (label2class1, label2class2)
    '''
    test_gen, steps, (label2class1, label2class2) = create_generator_gc(
        gc_path=gc_path,
        cities=cities,
        mode='test',
        batch_size=batch_size,
        random_state=random_state)

    X_test = np.empty((1, *input_shape))
    y1_test = np.empty((1, 2))
    y2_test = np.empty((1, len(cities)))
    for _ in range(steps):
        X_test_tmp, (y1_test_tmp, y2_test_tmp) = next(test_gen)
        X_test = np.vstack((X_test, X_test_tmp))
        y1_test = np.vstack((y1_test, y1_test_tmp))
        y2_test = np.vstack((y2_test, y2_test_tmp))

    X_test = X_test[1:]
    y1_test = y1_test[1:]
    y2_test = y2_test[1:]

    return X_test, (y1_test, y2_test), (label2class1, label2class2)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from load_data import load_processed_data, set_data

    dir_path = os.path.join('c:\\', 'Users', 'aviat', 'Google Drive', 'dl4us',
                            'prj')
    data = load_processed_data(dir_path, cities=('nyc', 'kyoto'), verbose=1)
    (X_train, X_valid,
     X_test), (y1_train, y1_valid,
               y1_test), (y2_train, y2_valid,
                          y2_test), (label2class1, label2class2) = set_data(
                              data, verbose=1)

    train_gen, steps_per_epoch = create_generator(
        X_train, (y1_train, y2_train), batch_size=32)

    X_gen, (y1_gen, y2_gen) = next(train_gen)

    we_city = zip(
        list(map(lambda i: label2class1[i][1], np.argmax(y1_gen, axis=1))),
        list(map(lambda i: label2class2[i][1], np.argmax(y2_gen, axis=1))))
    for i, (w_or_e, city) in enumerate(we_city):
        plt.imshow(X_gen[i])
        plt.title('{} - {}'.format(w_or_e, city))
        plt.show()
        if i == 4:
            break
