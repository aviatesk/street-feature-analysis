# -*- coding: utf-8 -*-
import os
import numpy as np
from load_model import _get_offtop
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


def extract(
        data,
        weights='places',
        pooling='avg',
        test_size=0.2,
        random_state=None,
        augment=False,
        augment_mode=0,
        augment_times=None,
        verbose=0,
):
    '''
    extract deep features from VGG16 pre-trained on Places365 dataset
    
    argments:
    - data: data used for prediction (should be a tuple whose shape is (city_name, city_data))
    - pooling: 'avg', 'max' (global pooling method used just after extracting the final deep features)
    - test_size: the ratio of train data vs. test data
    - random_state: the seed of train_test_split
    - augment: specifies whether augmentation applied to data or not
    - augment_mode: 0 will NOT apply augmentation to test data, whereas 1 will DO apply so
    - augment_times: specifies how many times I augment training data

    return: (extracted_data, original_data)
    - extracted_data: tuple whose element's shape is (city_name, (pld_train, pld_test), (npld_train, npld_test))
    - original_data: tuple whose element's shape is (city_name, (train_ret, test_ret))
    '''

    if not augment_times:
        augment_times = 1

    print('- test_size: {} - augment: {} - augment_mode: {} augment_times: {}'.
          format(test_size, augment, augment_mode, augment_times))

    # set models to extract
    pooling_model = _get_offtop(weights=weights, pooling=pooling)
    if not augment:
        model = _get_offtop(weights=weights)

    extracted_data = []
    original_data = []

    for city_name, city_data in data:

        train, test = train_test_split(
            city_data, test_size=test_size, random_state=random_state)

        # set generators
        if not augment:

            # set train generator
            train_gen = ImageDataGenerator(
                samplewise_center=True,
                samplewise_std_normalization=True,
            ).flow(
                train, shuffle=False)
            steps = len(train_gen) * augment_times
            # set test generator
            test_gen = ImageDataGenerator(
                samplewise_center=True,
                samplewise_std_normalization=True,
            ).flow(
                test, shuffle=False)

        else:

            if augment_mode:
                # set train generator
                train_gen = ImageDataGenerator(
                    samplewise_center=True,
                    samplewise_std_normalization=True,
                    rotation_range=20,
                    width_shift_range=0.15,
                    height_shift_range=0.15,
                    shear_range=0.15,
                    zoom_range=0.15,
                    horizontal_flip=True,
                ).flow(
                    train, shuffle=False)
                steps = len(
                    train_gen
                ) * augment_times  # training data will be doubled `augment_times` times
                # set test generator
                test_gen = ImageDataGenerator(
                    samplewise_center=True,
                    samplewise_std_normalization=True,
                    rotation_range=20,
                    width_shift_range=0.15,
                    height_shift_range=0.15,
                    shear_range=0.15,
                    zoom_range=0.15,
                    horizontal_flip=True,
                ).flow(
                    test, shuffle=False)

            else:
                # set train generator
                train_gen = ImageDataGenerator(
                    samplewise_center=True,
                    samplewise_std_normalization=True,
                    rotation_range=20,
                    width_shift_range=0.15,
                    height_shift_range=0.15,
                    shear_range=0.15,
                    zoom_range=0.15,
                    horizontal_flip=True,
                ).flow(
                    train, shuffle=False)
                steps = len(
                    train_gen
                ) * augment_times  # training data will be doubled `augment_times` times
                # set test generator
                test_gen = ImageDataGenerator(
                    samplewise_center=True,
                    samplewise_std_normalization=True,
                ).flow(
                    test, shuffle=False)

        print('extracting {} data ...'.format(city_name))

        # extract pooled features
        pld_train = pooling_model.predict_generator(
            generator=train_gen,
            steps=steps,
            verbose=verbose,
        )
        pld_test = pooling_model.predict_generator(
            generator=test_gen,
            verbose=verbose,
        )

        if not augment:
            # extract non flatten features
            npld_train = model.predict_generator(
                generator=train_gen,
                steps=steps,
                verbose=verbose,
            )
            npld_test = model.predict_generator(
                generator=test_gen,
                verbose=verbose,
            )

        # original data
        shape = train.shape
        train_ret = np.empty((augment_times, *shape))
        for i in range(augment_times):
            train_ret[i] = train
        test_ret = test

        if not augment:
            extracted_data.append((city_name, (pld_train, pld_test),
                                   (npld_train, npld_test)))
        else:
            extracted_data.append((city_name, (pld_train, pld_test)))

        original_data.append((city_name, train_ret, test_ret))

    if verbose:
        if not augment:
            print(
                'returned: (extracted_data, original_data)\n - extracted_data: {} length tuple whose element\'s shape is (city_name, (pld_train, pld_test), (npld_train, npld_test))\n - original_data: tuple whose element\'s shape is (city_name, (train_ret, test_ret))'
                .format(len(data)))
        else:
            print(
                'returned: (extracted_data, original_data)\n - extracted_data: {} length tuple whose element\'s shape is (city_name, (pld_train, pld_test), \n - original_data: tuple whose element\'s shape is (city_name, (train_ret, test_ret))'
                .format(len(data)))

    return extracted_data, original_data


def extract_gc(
        cities,
        gc_path='.',
        weights='places',
        pooling='avg',
        test_size=0.2,
        random_state=None,
        augment=False,
        augment_mode=0,
        augment_times=None,
        verbose=0,
):
    '''
    extract deep features from VGG16 pre-trained on Places365 dataset, in a way that does not use up memory and not return original_data
    does not take the original processed data but read it directly from virtual env in Google Colaboratory,
    thus assumes there are the processed files in the working directory in the env
    
    argments:
    - data: data used for prediction (should be a tuple whose shape is (city_name, city_data))
    - pooling: 'avg', 'max' (global pooling method used just after extracting the final deep features)
    - test_size: the ratio of train data vs. test data
    - random_state: the seed of train_test_split
    - augment: specifies whether augmentation applied to data or not
    - augment_mode: 0 will NOT apply augmentation to test data, whereas 1 will DO apply so
    - augment_times: specifies how many times I augment training data

    return: (extracted_data, original_data)
    - extracted_data: tuple whose element's shape is (city_name, (pld_train, pld_test), (npld_train, npld_test))
    '''

    if not augment_times:
        augment_times = 1

    print('- test_size: {} - augment: {} - augment_mode: {} augment_times: {}'.
          format(test_size, augment, augment_mode, augment_times))

    # set models to extract
    pooling_model = _get_offtop(weights=weights, pooling=pooling)
    if not augment:
        model = _get_offtop(weights=weights)

    extracted_data = []
    # original_data = []

    for city_name in cities:

        try:
            file_path = os.path.join(gc_path, city_name + '.npy')
            city_data = np.load(file_path)

        except FileNotFoundError:
            i = input(
                'file {} is not found, continue extracting the data without {}\'s data or not ? (y or n) >'
                .format(file_path, city_name))
            if i == 'y':
                print('continue extracting without {}\'s data ...'.format(
                    city_name))
                continue
            else:
                print('stopping  extracting ...')
                return extracted_data

        train, test = train_test_split(
            city_data, test_size=test_size, random_state=random_state)

        # set generators
        if not augment:

            # set train generator
            train_gen = ImageDataGenerator(
                samplewise_center=True,
                samplewise_std_normalization=True,
            ).flow(
                train, shuffle=False)
            steps = len(train_gen) * augment_times
            # set test generator
            test_gen = ImageDataGenerator(
                samplewise_center=True,
                samplewise_std_normalization=True,
            ).flow(
                test, shuffle=False)

        else:

            if augment_mode:
                # set train generator
                train_gen = ImageDataGenerator(
                    samplewise_center=True,
                    samplewise_std_normalization=True,
                    rotation_range=20,
                    width_shift_range=0.15,
                    height_shift_range=0.15,
                    shear_range=0.15,
                    zoom_range=0.15,
                    horizontal_flip=True,
                ).flow(
                    train, shuffle=False)
                steps = len(
                    train_gen
                ) * augment_times  # training data will be doubled `augment_times` times
                # set test generator
                test_gen = ImageDataGenerator(
                    samplewise_center=True,
                    samplewise_std_normalization=True,
                    rotation_range=20,
                    width_shift_range=0.15,
                    height_shift_range=0.15,
                    shear_range=0.15,
                    zoom_range=0.15,
                    horizontal_flip=True,
                ).flow(
                    test, shuffle=False)

            else:
                # set train generator
                train_gen = ImageDataGenerator(
                    samplewise_center=True,
                    samplewise_std_normalization=True,
                    rotation_range=20,
                    width_shift_range=0.15,
                    height_shift_range=0.15,
                    shear_range=0.15,
                    zoom_range=0.15,
                    horizontal_flip=True,
                ).flow(
                    train, shuffle=False)
                steps = len(
                    train_gen
                ) * augment_times  # training data will be doubled `augment_times` times
                # set test generator
                test_gen = ImageDataGenerator(
                    samplewise_center=True,
                    samplewise_std_normalization=True,
                ).flow(
                    test, shuffle=False)

        print('extracting {} data ...'.format(city_name))

        # extract pooled features
        pld_train = pooling_model.predict_generator(
            generator=train_gen,
            steps=steps,
            verbose=verbose,
        )
        pld_test = pooling_model.predict_generator(
            generator=test_gen,
            verbose=verbose,
        )

        if not augment:
            # extract non flatten features
            npld_train = model.predict_generator(
                generator=train_gen,
                steps=steps,
                verbose=verbose,
            )
            npld_test = model.predict_generator(
                generator=test_gen,
                verbose=verbose,
            )

        # # original data
        # shape = train.shape
        # train_ret = np.empty((augment_times, *shape))
        # for i in range(augment_times):
        #     train_ret[i] = train
        # test_ret = test

        if not augment:
            extracted_data.append((city_name, (pld_train, pld_test),
                                   (npld_train, npld_test)))
        else:
            extracted_data.append((city_name, (pld_train, pld_test)))

        # original_data.append((city_name, train_ret, test_ret))

    del city_data

    if verbose:
        if not augment:
            print(
                'returned: extracted_data\n - extracted_data: {} length tuple whose element\'s shape is (city_name, (pld_train, pld_test), (npld_train, npld_test))'
                .format(len(cities)))
        else:
            print(
                'returned: extracted_data\n - extracted_data: {} length tuple whose element\'s shape is (city_name, (pld_train, pld_test))'
                .format(len(cities)))

    return extracted_data  # original_data


if __name__ == '__main__':
    from load_data import load_processed_data

    dir_path = os.path.join('c:\\', 'Users', 'aviat', 'Google Drive', 'dl4us',
                            'prj')

    data = load_processed_data(dir_path, cities=('kyoto', ))
    tmp_data = (('kyoto', data[0][1][:10]), )

    extracted_data, original_data = extract(
        tmp_data,
        weights='places',
        pooling='avg',
        test_size=0.1,
        random_state=42,
        augment=True,
        augment_mode=0,
        augment_times=1,
    )
