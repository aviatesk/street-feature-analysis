# -*- coding: utf-8 -*-
import os
from keras.callbacks import ModelCheckpoint, CSVLogger


def SnapShot(dir_path, dir_name, model_name, monitor='val_acc', mode='max'):
    '''
    return keras.callbacks.Callback object which saves the model when `monitor` value improves and renews the best model

    argments:
    - monitor: the value to be monitored

    return: keras.callbacks.ModelCheckpoint object
    '''

    dirc_path = os.path.join(dir_path, 'models', dir_name)
    if not os.path.exists(dirc_path):
        os.mkdir(dirc_path)

    file_path = os.path.join(dirc_path, model_name) + '.best.h5'
    print('model snap shots will be saved as: {}'.format(file_path))

    ss = ModelCheckpoint(
        filepath=file_path,
        verbose=1,
        save_best_only=True,
        monitor=monitor,
        mode=mode)

    return ss


def Logger(dir_path, dir_name, base_name, add=False):
    '''
    return keras.callbacks.Callback object which saves the logs to `tmp_dir_path` through model fitting

    argments:
    - append: append if file exists (useful for continuing training) otherwise overwrite existing file

    return: keras.callbacks.CSVLogger object
    '''

    dirc_path = os.path.join(dir_path, 'hists', dir_name)
    if not os.path.exists(dirc_path):
        os.mkdir(dirc_path)
    file_path = os.path.join(dirc_path, base_name) + '.log.csv'
    print('fitting logs will be saved as: {}'.format(file_path))

    cl = CSVLogger(file_path, append=add)

    return cl
