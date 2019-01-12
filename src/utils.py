# -*- coding: utf-8 -*-
import os
import glob
import itertools
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def show_data(city_name, city_data, num_show=7):
    '''
    show images in city_data

    argments:
    - num_show: number of images shown in one figure
    '''

    print('{}: {}'.format(city_name, city_data.shape))

    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=0.05)

    for i in range(num_show):
        ax = fig.add_subplot(1, num_show, i + 1, xticks=[], yticks=[])
        ax.set_title(city_name)
        ax.imshow(city_data[i], cmap='gray')

    plt.show()


def plot_confusion_matrix(cm,
                          classes,
                          dir_path=None,
                          dir_name=None,
                          fig_name=None,
                          normalize=True,
                          title='confusion matrix',
                          cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.

    argments
    - classes: iterator containing class information of each labe (in order)
    - normalize: normalize the confusion matrix or not
    - title: title which will show in the generated figure
    '''

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if fig_name:
        if not dir_path:
            print('figure not saved because `dir_path` not specified')

        else:
            dirc_path = os.path.join(dir_path, 'figs', dir_name)
            if not os.path.exists(dirc_path):
                os.mkdir(dirc_path)
            file_path = os.path.join(dirc_path, fig_name + '.png')

            try:
                plt.savefig(file_path)
                print('figure saved successfully to {}'.format(file_path))
            except FileNotFoundError:
                print(
                    'figure not saved successfully because the file path {} does not work, check whether the target directory exists'
                    .format(file_path))

    plt.show()


def report(y_true,
           y_pred,
           mode='train',
           classes=None,
           dir_path=None,
           dir_name=None,
           fig_name=None,
           normalize=True,
           digits=4):
    '''
    report the metrics and confusion matrix against the given labels
    note the order of labels

    argments
    - mode: specifies training or validation or test, given as str ('train', 'validation', 'test') or integer (0: 'train', 1: 'validation', 2: 'test')
    - classes: iterator containing class information of each labe (in order)
    - normalize: normalize the confusion matrix or not
    - digits: number of digits for formatting output floating point values
    '''

    mode = 'train' if mode == 0 else\
           'valid' if mode == 1 else\
           'test' if mode == 2 else\
           mode

    print('----- {} -----'.format(mode))
    accuracy = accuracy_score(y_true, y_pred)
    print('accuracy: {:.4f} %'.format(accuracy))
    print(
        classification_report(
            y_true, y_pred, digits=digits, target_names=classes))

    if mode is not 'train':
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(
            cm,
            classes=classes,
            normalize=normalize,
            title='{} accuracy: {:.4f}'.format(mode, accuracy),
            dir_path=dir_path,
            dir_name=dir_name,
            fig_name=fig_name,
        )


def save_histories(
        dir_path,
        dir_name,
        hist_name,
        history,
        add=False,
):
    '''
    take the keras.callbacks.History object and save its logs into specified directory

    argments:
    add: add history to the already existing files
    '''

    dirc_path = os.path.join(dir_path, 'hists', dir_name)
    if not os.path.exists(dirc_path):
        os.makedirs(dirc_path)
    file_path_base = os.path.join(dirc_path, hist_name)

    for k, v in history.history.items():
        if not add:
            hist = pd.Series(v, name=k)
            hist.to_csv(
                file_path_base + '_' + k + '.csv',
                header=True,
                index_label='epoch',
            )

        else:
            old_hist = pd.read_csv(file_path_base + '_' + k + '.csv')[k]
            new_hist = pd.Series(v, name=k)
            hist = pd.concat([old_hist, new_hist])
            hist.reset_index(inplace=True, drop=True)
            hist.to_csv(
                file_path_base + '_' + k + '.csv',
                header=True,
                index_label='epoch',
            )

    print('all history files saved successfully to {}'.format(dirc_path))


def save_model(
        dir_path,
        dir_name,
        model_name,
        model,
):
    '''
    take keras.models.Model object and save it into specified directory
    '''

    dirc_path = os.path.join(dir_path, 'models', dir_name)
    if not os.path.exists(dirc_path):
        os.mkdirs(dirc_path)
    file_path = os.path.join(dirc_path, model_name)

    model.save(file_path + '.h5')
    print('model h5 file saved successfully as {}'.format(file_path))


def show_hist_from_log(
        dir_path,
        dir_name,
        base_names=None,
        hists=('output1_acc', 'output2_acc', 'val_output1_acc',
               'val_output2_acc'),
        fig_name=None,
):
    '''
    show the various history values specified in a single figure

    argments:
    - hists: tuple of hist names
    - base_names: tuple of the base model names or None, if None log files will be searched by `glob.glob(os.path.join(dirc_path, '*')`
    - fig_name: save the figure if specified using itself as the file name
    '''

    dirc_path = os.path.join(dir_path, 'hists', dir_name)

    if base_names:
        log_files = [
            os.path.join(dirc_path, base_name + '.log.csv')
            for base_name in base_names
        ]
    else:
        log_files = glob.glob(os.path.join(dirc_path, '*' + '.log.csv'))

    for f in log_files:
        d, n = os.path.split(f)
        table = pd.read_csv(f)
        for hist in hists:
            plt.plot(table[hist], label=n[:-8] + ' : ' + hist)
        plt.xlabel('epochs')
        plt.ylabel(hists[0].split('_')[-1])
    plt.legend()
    plt.title('{} - {}'.format(dir_name, base_names))

    if fig_name:
        dirc_path = os.path.join(dir_path, 'figs', dir_name)
        if not os.path.exists(dirc_path):
            os.makedirs(dirc_path)

        file_path = os.path.join(dirc_path, fig_name)
        try:
            plt.savefig(file_path)
            print('history figure saved successfully as {}'.format(file_path))

        except FileNotFoundError:
            print(
                'figure not saved successfully because the file path {} does not work'
                .format(file_path))
    plt.show()


def show_cv_img(cv_img):
    '''
    show an image opened with OpenCV in matplotlib
    '''

    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    dir_path = os.path.join('c:\\', 'Users', 'aviat', 'Google Drive', 'dl4us',
                            'prj')
    dir_name = 'fine-tuned'
