# -*- coding: utf-8 -*-
import os
import sys
import requests
from datetime import datetime
from urllib.parse import urlparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DIR_PATH = os.path.join('.')
DEFAULT_IMAGE_PATH = os.path.join(DIR_PATH, 'data', 'downloaded', 'demo.jpg')
DEFAULT_MODEL_FILE_PATH = os.path.join(DIR_PATH, 'models',
                                       '512-avg-0.5vs1-15.best.default.h5')
DEFAULT_SAVE_DIR = os.path.join(DIR_PATH, 'results')
DEFAULT_DONWLOAD_DIR = os.path.join(DIR_PATH, 'data', 'downloaded')
DEFAULT_LOCALIZED_CLS2 = 0
ONLY_PREDICTION_FLAG = False
ONLY_CAM_FLAG = False
VISUALIZE_FLAG = True
SAVE_FLAG = True
VERBOSE = 1
INFO_FUNC = lambda: print('[INFO]:', end=' ')


def download_img(url, save_name, print_func=None, verbose=1):

    if print_func:
        print_func()
    print('Downloading image from {} ...'.format(url))

    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(save_name, 'wb') as f:
            f.write(r.content)

        if verbose:
            if print_func:
                print_func()
            print('Downloaded image was saved as {}'.format(save_name))

        return save_name

    else:
        INFO_FUNC()
        print('Failed to save an image from {}, skip processing'.format(url))
        return False


if __name__ == '__main__':

    image_paths = []  # list of (localized_cls2, save_flag, image_path)
    display_cnt = 0
    print_func = lambda: print('[Process {}]:'.format(display_cnt), end=' ')

    argv = sys.argv[1:]
    if not argv:
        print_func()
        print('There is no input, thus let\'s go Demo run with {}'.format(
            DEFAULT_IMAGE_PATH))
        image_paths.append(
            [DEFAULT_LOCALIZED_CLS2, 'demo', DEFAULT_IMAGE_PATH])

    elif argv[0] == '-h':
        print(
            '> python run.py IMAGEPATH [IMAGEPATH -c CLSNUM] [IMAGEPATH -n SAVENAME] [IMAGEPATH --no-save] [-d SAVEDIRECTORY] [-m MODELFILEPATH] [--only-prediction] [--only-cam] [--no-visualize] [--no-save-all] [--no-verbose]'
        )
        print(
            '\n## This program runs Street10-CNN classsifying input image(s) and then show attributions for the result, using Guided-GradCAM method'
        )
        print(
            ' - IMAGEPATH can be URL to an image on the web, which will be loaded with requests module'
        )
        print(' - Downloaded images will be saved into {} by default'.format(
            DEFAULT_DONWLOAD_DIR))
        print(' - Result figures will be saved into {} by default'.format(
            DEFAULT_SAVE_DIR))
        print(' - {} will be used as the default Street10-CNN'.format(
            DEFAULT_MODEL_FILE_PATH))

        print('\n## Command options')
        print(
            ' - [IMAGEPATH -c CLSNUM]: saliency for class CLSNUM in output2 will be localized'
        )
        print(
            ' - [IMAGEPATH -n SAVENAME]: Figure will be saved as SAVENAME (.png format as default)'
        )
        print(' - [IMAGEPATH --no-save]: Figure will not be saved')
        print(
            ' - [-d SAVEDIRECTORY]: All the figures will be saved in to SAVEDIRECTORY'
        )
        print(
            ' - [-m MODELFILEPATH]: MODELFILEPATH (should be hdf5 format) will be used as Street10-CNN'
        )
        print(' - [--only-prediction]: Run only predictions')
        print(
            ' - [--only-cam]: Run only CAM after making predication (the default size of image will be used for the result figures)'
        )
        print(' - [--no-visualize]: All the figures will not be shown')
        print(' - [--no-save-all]: All the figures will not be saved')
        print(' - [--no-verbose]: Run in non-verbose mode')
        sys.exit(0)

    else:
        skip = False
        for i, cmd in enumerate(argv):
            if skip:
                skip = False
                continue

            elif cmd == '-c':
                image_paths[-1][0] = int(argv[i + 1])
                skip = True
                print_func()
                print(
                    'The class {} lower from the highest class will be localized for output2'
                    .format(image_paths[-1][0]))

            elif cmd == '-n':
                image_paths[-1][1] = argv[i + 1]
                skip = True
                print_func()
                print('The result image will be saved as {}.png'.format(
                    image_paths[-1][1]))

            elif cmd == '--no-save':
                image_paths[-1][1] = 0
                print_func()
                print('The result image will NOT be saved')

            elif cmd == '-d':
                DEFAULT_SAVE_DIR = argv[i + 1]
                skip = True
                INFO_FUNC()
                print('All the figures will be saved in {}'.format(
                    DEFAULT_SAVE_DIR))

            elif cmd == '-m':
                model_file_path = argv[i + 1]
                skip = True
                INFO_FUNC()
                print('{} will be used for prediction'.format(model_file_path))

            elif cmd == '--only-prediction':
                ONLY_PREDICTION_FLAG = True
                INFO_FUNC()
                print('Run only predictions')

            elif cmd == '--only-cam':
                ONLY_CAM_FLAG = True
                INFO_FUNC()
                print('Only CAM method will be used for explanations')

            elif cmd == '--no-visualize':
                VISUALIZE_FLAG = False
                INFO_FUNC()
                print('All figures will not be showed')
                if not SAVE_FLAG:
                    print('[CAUSION]: All figures will not be saved too')

            elif cmd == '--no-save-all':
                SAVE_FLAG = False
                INFO_FUNC()
                print('All figures will not be saved')

            elif cmd == '--no-verbose':
                VERBOSE = 0

            else:
                display_cnt += 1
                save_flag = datetime.now().isoformat().replace(
                    ':', '-').replace('.', '-') + str(display_cnt)
                if len(urlparse(cmd).scheme) > 0:
                    url = cmd
                    save_name = os.path.join(
                        DEFAULT_DONWLOAD_DIR,
                        save_flag.replace(':', '-') + '.jpg')
                    image_path = download_img(
                        url, save_name, print_func=print_func, verbose=VERBOSE)
                    if not image_path:
                        display_cnt -= 1
                        continue

                    ONLY_CAM_FLAG = True

                else:
                    image_path = cmd

                image_paths.append([
                    DEFAULT_LOCALIZED_CLS2,
                    save_flag,
                    image_path,
                ])

    from keras.models import load_model
    from src.guided_grad_cam import set_guided_model, show_saliency

    # load model
    INFO_FUNC()
    try:
        print('Loading {} ...'.format(model_file_path))

    except NameError:
        model_file_path = DEFAULT_MODEL_FILE_PATH
        print('Loading {} ...'.format(model_file_path))

    except FileNotFoundError:
        print('{} not found, loading {} ...'.format(model_file_path,
                                                    DEFAULT_MODEL_FILE_PATH))
        model_file_path = DEFAULT_MODEL_FILE_PATH

    model = load_model(model_file_path)
    guided_model = set_guided_model(model_file_path)

    # move to main process
    for i, (localized_cls2, tmp_save_flag,
            image_path) in enumerate(image_paths):
        display_cnt = i + 1
        tmp_save = False if not SAVE_FLAG else \
                   False if not tmp_save_flag else \
                   True

        print()
        show_saliency(
            model,
            guided_model,
            image_path,
            localized_cls2=localized_cls2,
            only_prediction=ONLY_PREDICTION_FLAG,
            only_cam=ONLY_CAM_FLAG,
            visualize=VISUALIZE_FLAG,
            save=tmp_save,
            save_dir=DEFAULT_SAVE_DIR,
            save_name=tmp_save_flag,
            print_func=print_func,
            verbose=VERBOSE,
        )
