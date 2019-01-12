# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import show_data

H, W, C = 224, 224, 3
WESTERNS = ('london', 'moscow', 'nyc', 'paris', 'vancouver')
EASTERNS = ('beijing', 'kyoto', 'seoul', 'singapore', 'tokyo')


def load_processed_data(
        dir_path,
        cities=('london', 'moscow', 'nyc', 'paris', 'vancouver', 'beijing',
                'kyoto', 'seoul', 'singapore', 'tokyo'),
        input_shape=(H, W, C),
        show=None,
        verbose=0,
):
    '''
    load a processed data from each jpg files and show it if specified

    argments
    - gc: used in Google Colaboratory

    return: tuple whose element's shape is (city_name, city_data)
    - city_name: string
    - city_data: numpy array
    '''

    data = []

    for city_name in cities:

        if verbose:
            print('loading {} street data ...'.format(city_name))

        city_dirc_path = os.path.join(dir_path, 'data', 'processed', '*',
                                      city_name, '*')
        dirc_paths = glob.glob(city_dirc_path)

        if not dirc_paths:
            i = input(
                'city_dirc_path {} does not seem to exist, continue loading the data without {}\'s data or not ? (y or n) >'
                .format(city_dirc_path, city_name))
            if i == 'y':
                print('continue loading without {}\'s data ...'.format(
                    city_name))
                continue
            else:
                print('stopping loading ...')
                return data

        tmp_data = np.empty((1, *input_shape), dtype='float')
        for dirc_path in dirc_paths:

            files = glob.glob(os.path.join(dirc_path, '*.jpg'))
            _ = np.empty((len(files), *input_shape), dtype='float')

            if verbose:
                for i, f in tqdm(enumerate(files)):
                    _[i] = np.asarray(Image.open(f)) / 255.
            else:
                for i, f in enumerate(files):
                    _[i] = np.asarray(Image.open(f)) / 255.

            tmp_data = np.vstack((tmp_data, _))

        data.append((city_name, tmp_data[1:]))

    if show:

        for city_name, city_data in data:
            show_data(city_name, city_data, num_show=show)

    if verbose:
        print(
            'returned: {} length tuple whose element\'s shape is (city_name, city_data)'
            .format(len(data)))
        for city_name, city_data in data:
            print('{}: {}'.format(city_name, city_data.shape))

    return data


def load_processed_data_gc(
        gc_path='.',
        cities=('london', 'moscow', 'nyc', 'paris', 'vancouver', 'beijing',
                'kyoto', 'seoul', 'singapore', 'tokyo'),
        input_shape=(H, W, C),
        show=None,
        verbose=0,
):
    '''
    load a processed data from npy files laid in virtual env in Google Colaboratory, and show it if specified
    thus assumes there are the processed files in the working directory in the env

    argments
    - show: None or int (which specifies the number of images of each city will be showed)

    return: tuple whose element's shape is (city_name, city_data)
    - city_name: string
    - city_data: numpy array
    '''

    data = []

    for city_name in cities:

        if verbose:
            print('loading {} street data ...'.format(city_name))

        try:
            file_path = os.path.join(gc_path, city_name + '.npy')
            tmp_data = np.load(file_path)

        except FileNotFoundError:
            i = input(
                'file {} is not found, continue loading the data without {}\'s data or not ? (y or n) >'
                .format(file_path, city_name))
            if i == 'y':
                print('continue loading without {}\'s data ...'.format(
                    city_name))
                continue
            else:
                print('stopping loading ...')
                return data

        data.append((city_name, tmp_data))

    if show:

        for city_name, city_data in data:
            show_data(city_name, city_data, num_show=show)

    if verbose:
        print(
            'returned: {} length tuple whose element\'s shape is (city_name, city_data)'
            .format(len(data)))
        for city_name, city_data in data:
            print('{}: {}'.format(city_name, city_data.shape))

    return data


def set_extracted_data(
        extracted_data,
        cities=('london', 'moscow', 'nyc', 'paris', 'vancouver', 'beijing',
                'kyoto', 'seoul', 'singapore', 'tokyo'),
        verbose=0,
):
    '''
    set X and y from extracted deep-features data extracted from Places365-VGG16, which are assumed to be used with other classifiers, like SVM and Logistic Regression.

    return: (X_pld_train, X_pld_test), (X_npld_train, X_npld_test), (y1_train, y1_test), (y2_train, y2_test), (label2class1, label2class2)
    - X_pld_train: pooled training data
    - X_pld_test: pooled test data
    - X_npld_train: non-pooled training data
    - X_npld_test: non-pooled test data
    - y1_train: training target label representing a training sample is Western(0) or Eastern(1)
    - y1_test: test target lable representing a test sample is Western(0) or Eastern(1)
    - y2_train: training target label representing a training sample's city
    - y2_test: test target label representing a test sample's city
    - label2class1: label information about Western or Eastern
    - label2class2: label information about city names
    '''

    if len(extracted_data[0]) == 2:
        augmented = True
    else:
        augmented = False

    _, (pld, _), *npld = extracted_data[0]
    shape1 = pld.shape
    if not augmented:
        shape2 = npld[0][0].shape

    # X
    # pooled data
    X_pld_train = np.empty((1, *shape1[1:]))
    X_pld_test = np.empty((1, *shape1[1:]))
    if not augmented:
        # non pooled data
        X_npld_train = np.empty((1, *shape2[1:]))
        X_npld_test = np.empty((1, *shape2[1:]))

    # y
    y1_train = np.empty(1, dtype='int')
    y2_train = np.empty(1, dtype='int')
    y1_test = np.empty(1, dtype='int')
    y2_test = np.empty(1, dtype='int')

    # label data
    label = 0
    label2class1 = [(0, 'western'), (1, 'eastern')]
    label2class2 = []

    if not augmented:
        for city_name, (pld_train, pld_test), (npld_train,
                                               npld_test) in extracted_data:
            if city_name not in cities:
                continue

            X_pld_train = np.vstack((X_pld_train, pld_train))
            X_pld_test = np.vstack((X_pld_test, pld_test))

            X_npld_train = np.vstack((X_npld_train, npld_train))
            X_npld_test = np.vstack((X_npld_test, npld_test))

            train_size = pld_train.shape[0]
            test_size = pld_test.shape[0]

            y1_train_tmp = np.zeros(
                train_size, dtype='int') if city_name in WESTERNS else np.ones(
                    train_size, dtype='int')
            y1_train = np.hstack((y1_train, y1_train_tmp))
            y1_test_tmp = np.zeros(
                test_size, dtype='int') if city_name in WESTERNS else np.ones(
                    test_size, dtype='int')
            y1_test = np.hstack((y1_test, y1_test_tmp))

            y2_train_tmp = np.ones(train_size).astype('int') * label
            y2_train = np.hstack((y2_train, y2_train_tmp))
            y2_test_tmp = np.ones(test_size).astype('int') * label
            y2_test = np.hstack((y2_test, y2_test_tmp))

            label2class2.append((label, city_name))
            label += 1

    else:
        for city_name, (pld_train, pld_test) in extracted_data:
            if city_name not in cities:
                continue

            X_pld_train = np.vstack((X_pld_train, pld_train))
            X_pld_test = np.vstack((X_pld_test, pld_test))

            train_size = pld_train.shape[0]
            test_size = pld_test.shape[0]

            y1_train_tmp = np.zeros(
                train_size, dtype='int') if city_name in WESTERNS else np.ones(
                    train_size, dtype='int')
            y1_train = np.hstack((y1_train, y1_train_tmp))
            y1_test_tmp = np.zeros(
                test_size, dtype='int') if city_name in WESTERNS else np.ones(
                    test_size, dtype='int')
            y1_test = np.hstack((y1_test, y1_test_tmp))

            y2_train_tmp = np.ones(train_size).astype('int') * label
            y2_train = np.hstack((y2_train, y2_train_tmp))
            y2_test_tmp = np.ones(test_size).astype('int') * label
            y2_test = np.hstack((y2_test, y2_test_tmp))

            label2class2.append((label, city_name))
            label += 1

    # pooled data
    X_pld_train = X_pld_train[1:]
    X_pld_test = X_pld_test[1:]
    # standarize
    sc_pld = StandardScaler()
    sc_pld.fit(X_pld_train)
    X_pld_train = sc_pld.transform(X_pld_train)
    X_pld_test = sc_pld.transform(X_pld_test)

    if not augmented:
        # non pooled data
        X_npld_train = X_npld_train[1:]
        X_npld_test = X_npld_test[1:]
        # flatten
        _, h, w, c = X_npld_train.shape
        X_npld_train = X_npld_train.reshape(-1, h * w * c)
        X_npld_test = X_npld_test.reshape(-1, h * w * c)
        # standarize
        sc = StandardScaler()
        sc.fit(X_npld_train)
        X_npld_train = sc.transform(X_npld_train)
        X_npld_test = sc.transform(X_npld_test)

    # target data
    y1_train = y1_train[1:]
    y1_test = y1_test[1:]
    y2_train = y2_train[1:]
    y2_test = y2_test[1:]

    if not augmented:
        if verbose:
            print(
                'returned: (X_pld_train, X_pld_test), (X_npld_train, X_npld_test), (y1_train, y1_test), (y2_train, y2_test), (label2class1, label2class2)'
            )
            print('X_pld_train:', X_pld_train.shape)
            print('X_pld_test:', X_pld_test.shape)
            print('X_npld_train:', X_npld_train.shape)
            print('X_npld_test:', X_npld_test.shape)
            print('y1_train:', y1_train.shape)
            print('y1_test:', y1_test.shape)
            print('y2_train:', y2_train.shape)
            print('y2_test:', y2_test.shape)
            print('label2class1:', label2class1)
            print('label2class2:', label2class2)

        return (X_pld_train, X_pld_test), (X_npld_train, X_npld_test), (
            y1_train, y1_test), (y2_train, y2_test), (label2class1,
                                                      label2class2)
    else:
        if verbose:
            print(
                'returned: (X_pld_train, X_pld_test), (y1_train, y1_test), (y2_train, y2_test), (label2class1, label2class2)'
            )
            print('X_pld_train:', X_pld_train.shape)
            print('X_pld_test:', X_pld_test.shape)
            print('y1_train:', y1_train.shape)
            print('y1_test:', y1_test.shape)
            print('y2_train:', y2_train.shape)
            print('y2_test:', y2_test.shape)
            print('label2class1:', label2class1)
            print('label2class2:', label2class2)

    return (X_pld_train, X_pld_test), (y1_train,
                                       y1_test), (y2_train,
                                                  y2_test), (label2class1,
                                                             label2class2)


def set_data(
        data,
        cities=('london', 'moscow', 'nyc', 'paris', 'vancouver', 'beijing',
                'kyoto', 'seoul', 'singapore', 'tokyo'),
        test_size=0.1,
        validation_size=0.15,
        random_state=42,
        verbose=0,
):
    '''
    set X and y for CNN architecture network

    augments
    - test_size: the ratio of train and validation data vs. test data
    - validation_size: the ratio of train data vs. validation data
    - random_state: the seed of train_test_split

    return: (X_train, X_valid, X_test), (y1_train, y1_valid, y1_test), (y2_train, y2_valid, y2_test), (label2class1, label2class2)
    - X_train: training data
    - X_valid: validation data
    - X_test: test data
    - y1_train: training target label representing a training sample is Western(0) or Eastern(1)
    - y1_valid: validation target label representing a training sample is Western(0) or Eastern(1)
    - y1_test: test target lable representing a test sample is Western(0) or Eastern(1)
    - y2_train: training target label representing a training sample's city
    - y2_valid: validation target label representing a training sample's city
    - y2_test: test target label representing a test sample's city
    - label2class1: label information about Western or Eastern
    - label2class2: label information about city names
    '''

    _, city_data = data[0]
    input_shape = city_data.shape[1:]

    # X
    X_train = np.empty((1, *input_shape))
    X_valid = np.empty((1, *input_shape))
    X_test = np.empty((1, *input_shape))

    # y
    y1_train = np.empty(1)
    y1_valid = np.empty(1)
    y1_test = np.empty(1)
    y2_train = np.empty(1)
    y2_valid = np.empty(1)
    y2_test = np.empty(1)

    # label data
    label = 0
    label2class1 = [(0, 'western'), (1, 'eastern')]
    label2class2 = []

    for city_name, city_data in data:

        if city_name not in cities:
            continue

        train, test = train_test_split(
            city_data, test_size=test_size, random_state=random_state)
        train, valid = train_test_split(
            train, test_size=validation_size, random_state=random_state)
        _train_size = train.shape[0]
        _valid_size = valid.shape[0]
        _test_size = test.shape[0]

        X_train = np.vstack((X_train, train))
        X_valid = np.vstack((X_valid, valid))
        X_test = np.vstack((X_test, test))

        y1_train_tmp = np.zeros(
            _train_size) if city_name in WESTERNS else np.ones(_train_size)
        y1_train = np.hstack((y1_train, y1_train_tmp))
        y1_valid_tmp = np.zeros(
            _valid_size) if city_name in WESTERNS else np.ones(_valid_size)
        y1_valid = np.hstack((y1_valid, y1_valid_tmp))
        y1_test_tmp = np.zeros(
            _test_size) if city_name in WESTERNS else np.ones(_test_size)
        y1_test = np.hstack((y1_test, y1_test_tmp))

        y2_train_tmp = np.ones(_train_size) * label
        y2_train = np.hstack((y2_train, y2_train_tmp))
        y2_valid_tmp = np.ones(_valid_size) * label
        y2_valid = np.hstack((y2_valid, y2_valid_tmp))
        y2_test_tmp = np.ones(_test_size) * label
        y2_test = np.hstack((y2_test, y2_test_tmp))

        label2class2.append((label, city_name))
        label += 1

    X_train = X_train[1:]
    X_valid = X_valid[1:]
    X_test = X_test[1:]
    y1_train = y1_train[1:]
    y1_valid = y1_valid[1:]
    y1_test = y1_test[1:]
    y2_train = y2_train[1:]
    y2_valid = y2_valid[1:]
    y2_test = y2_test[1:]

    # make label data into categorical variables
    y1_train = to_categorical(y1_train)
    y1_valid = to_categorical(y1_valid)
    y1_test = to_categorical(y1_test)
    y2_train = to_categorical(y2_train)
    y2_valid = to_categorical(y2_valid)
    y2_test = to_categorical(y2_test)

    if verbose:
        print(
            'returned: (X_train, X_valid, X_test), (y1_train, y1_valid, y1_test), (y2_train, y2_valid, y2_test), (label2class1, label2class2)'
        )
        print('X_train:', X_train.shape)
        print('X_valid:', X_valid.shape)
        print('X_test:', X_test.shape)
        print('y1_train:', y1_train.shape)
        print('y1_valid:', y1_valid.shape)
        print('y1_test:', y1_test.shape)
        print('y2_train:', y2_train.shape)
        print('y2_valid:', y2_valid.shape)
        print('y2_test:', y2_test.shape)
        print('label2class1:', label2class1)
        print('label2class2:', label2class2)

    return (X_train, X_valid, X_test), (y1_train, y1_valid,
                                        y1_test), (y2_train, y2_valid,
                                                   y2_test), (label2class1,
                                                              label2class2)


def set_data_gc(
        gc_path='.',
        cities=('london', 'moscow', 'nyc', 'paris', 'vancouver', 'beijing',
                'kyoto', 'seoul', 'singapore', 'tokyo'),
        input_shape=(H, W, C),
        test_size=0.1,
        validation_size=0.15,
        random_state=42,
        verbose=0,
):
    '''
    create the directory containing X data for CNN architecture network, which should be used with keras.preprocessing.image.ImageDataGenerator.flow_from_directory
    does not take the original processed data but read it directly from virtual env in Google Colaboratory,
    thus assumes there are the processed files in the working directory in the env

    augments
    - input_shape: tuple representing the original processed image's shape
    - test_size: the ratio of train and validation data vs. test data
    - validation_size: the ratio of train data vs. validation data
    - random_state: the seed of train_test_split

    '''

    dirc_path = os.path.join(gc_path, 'data')

    for city_name in tqdm(cities):

        train_path = os.path.join(dirc_path, 'train', city_name)
        if not os.path.exists(train_path):
            os.makedirs(train_path)

        validate_path = os.path.join(dirc_path, 'validate', city_name)
        if not os.path.exists(validate_path):
            os.makedirs(validate_path)

        test_path = os.path.join(dirc_path, 'test', city_name)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        try:
            file_path = os.path.join(gc_path, city_name + '.npy')
            city_data = np.load(file_path)

        except FileNotFoundError:
            i = input(
                '\nfile {} is not found, continue setting the data without {}\'s data or not ? (y or n) > '
                .format(file_path, city_name))
            if i == 'y':
                print('continue setting without {}\'s data ...'.format(
                    city_name))
                continue
            else:
                print('stopping setting the data ...')
                return

        train, test = train_test_split(
            city_data, test_size=test_size, random_state=random_state)
        train, validation = train_test_split(
            train, test_size=validation_size, random_state=random_state)

        if verbose:
            print('\nprocessing {} training data ...'.format(city_name))
        for i, data in enumerate(train):
            path = os.path.join(train_path, str(i) + '.png')
            Image.fromarray(np.uint8(data * 255)).save(path)

        if verbose:
            print('processing {} validation data ...'.format(city_name))
        for i, data in enumerate(validation):
            path = os.path.join(validate_path, str(i) + '.png')
            Image.fromarray(np.uint8(data * 255)).save(path)

        if verbose:
            print('processing {} test data ...'.format(city_name))
        for i, data in enumerate(test):
            path = os.path.join(test_path, str(i) + '.png')
            Image.fromarray(np.uint8(data * 255)).save(path)

    del city_data

    print('\nall data saving has been done successfully')


if __name__ == '__main__':
    from extract import extract

    dir_path = os.path.join('c:\\', 'Users', 'aviat', 'Google Drive', 'dl4us',
                            'prj')

    data = load_processed_data(dir_path, cities=('nyc', 'kyoto'), verbose=1)
    tmp_data = (('kyoto', data[0][1][:10]), ('tokyo', data[1][1][:10]))

    extracted_data, original_data = extract(
        tmp_data,
        weights='places',
        pooling='avg',
        test_size=0.2,
        random_state=42,
        augment=False,
        augment_mode=0,
        augment_times=1,
    )

    (X_pld_train,
     X_pld_test), (X_npld_train, X_npld_test), (y1_train, y1_test), (
         y2_train, y2_test), (label2class1, label2class2) = set_extracted_data(
             extracted_data, verbose=1)

    (X_train, X_valid,
     X_test), (y1_train, y1_valid,
               y1_test), (y2_train, y2_valid,
                          y2_test), (label2class1, label2class2) = set_data(
                              data, verbose=1)
