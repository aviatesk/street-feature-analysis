# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


def process(
        dir_path,
        dirc_path,
        timestamp,
        command,
        keyword=None,
        crop_rate=0.9,
        image_size=(224, 224, 3),
        threshold=0.999,
        verbose=0,
):
    '''
    crop, resize, and then detect and pass a duplicated image
    '''

    _, city_name = os.path.split(dirc_path)
    _, super_concept = os.path.split(_)

    input_dir = os.path.join(dir_path, 'data', 'original', super_concept,
                             city_name)
    if not os.path.exists(input_dir):
        raise OSError('`input_dir` {} not found'.format(input_dir))

    output_dir = os.path.join(dir_path, 'data', 'processed', super_concept,
                              city_name)

    dirc_paths = glob.glob(os.path.join(input_dir, '*'))
    if command == 'add':
        tmp_path = os.path.join(input_dir, keyword)
        dirc_paths.remove(tmp_path)
        dirc_paths.append(tmp_path)
        add_cnt1 = len(glob.glob(os.path.join(dirc_paths[-1], '*')))
        add_cnt2 = 0

    hists = []
    all_cnt1 = 0
    all_cnt2 = 0
    print('\nprocessed file will be saved in the directory {}'.format(
        output_dir))
    print(' - crop_rate: {} - image_size: {} - threshold: {}'.format(
        crop_rate, image_size, threshold))

    for dirc_path in dirc_paths:

        _, tmp_keyword = os.path.split(dirc_path)
        tmp_output_path = os.path.join(output_dir,
                                       tmp_keyword + '_' + timestamp)
        if not os.path.exists(tmp_output_path):
            os.makedirs(tmp_output_path)

        files = glob.glob(os.path.join(dirc_path, '*'))
        cnt = 0
        all_cnt1 += len(files)

        for f in tqdm(files):
            img = cv2.imread(f)

            # change format if needed
            if img is None:
                print('\nthe file converted to .jpg format for processing: {}'.
                      format(f))
                tmp_img = Image.open(f, 'r')
                canvas = Image.new('RGB', tmp_img.size, (255, 255, 255))
                canvas.paste(tmp_img, (0, 0))
                img = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

            h, w, c = img.shape

            # crop marginal areas
            hcs = int(h * (1 - crop_rate) / 2)
            wcs = int(w * (1 - crop_rate) / 2)
            img = img[hcs:h - hcs, wcs:w - wcs]

            # resize
            img = cv2.resize(img, image_size[:-1])

            # delete duplicated images
            # save RGB histgram
            tmp_hist = []
            tmp_hist.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
            tmp_hist.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
            tmp_hist.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
            # compare to histgrams of already passed images
            for hist, f_hist in hists:
                if (cv2.compareHist(tmp_hist[0], hist[0], cv2.HISTCMP_CORREL) > threshold) \
                   or (cv2.compareHist(tmp_hist[1], hist[1], cv2.HISTCMP_CORREL) > threshold) \
                   or (cv2.compareHist(tmp_hist[2], hist[2], cv2.HISTCMP_CORREL) > threshold):
                    print('\nthe duplicated file passed: {} <===> {}'.format(
                        f_hist, f))
                    break

            else:
                hists.append((tmp_hist, f))
                # save
                if command == 'all':
                    cv2.imwrite(
                        os.path.join(tmp_output_path,
                                     str(cnt + 1) + '.jpg'), img)
                if command == 'add' and tmp_keyword == keyword:
                    cv2.imwrite(
                        os.path.join(tmp_output_path,
                                     str(cnt + 1) + '.jpg'), img)
                    add_cnt2 += 1

                cnt += 1
                all_cnt2 += 1

        if command == 'add' and tmp_keyword == keyword:
            # report
            print('the preprocessing suceeded!:', city_name, keyword)
            print('the number of original images:', add_cnt1)
            print('the number of processed images:', add_cnt2)
            return

    # report
    print('the preprocessing suceeded!:', city_name)
    print('the number of original images:', all_cnt1)
    print('the number of processed images:', all_cnt2)
    return


if __name__ == '__main__':
    import sys

    dir_path = os.path.join('c:\\', 'Users', 'aviat', 'Google Drive', 'dl4us',
                            'prj')
    western_cities = ('london', 'moscow', 'nyc', 'paris', 'vancouver')
    eastern_cities = ('beijing', 'kyoto', 'seoul', 'singapore', 'tokyo')

    argv = sys.argv
    try:
        timestamp, command = argv[1:3]
    except ValueError:
        print(
            'preprocess.py should be called at least two argments > python preprocess.py timestamp command'
        )
        exit()

    if command == 'all':
        dirc_paths = glob.glob(
            os.path.join(dir_path, 'data', 'original', '*', '*'))

        for dirc_path in dirc_paths:
            process(dir_path, dirc_path, timestamp, command)

    elif command == 'add':
        try:
            city_name, keyword = argv[3:5]
        except ValueError:
            print(
                'if `command` is add, you should add two more argment > python preprocess.py timestamp add city_name keyword'
            )
            exit()

        w_or_e = 'western' if city_name in western_cities else 'eastern'
        dirc_path = os.path.join(dir_path, 'data', 'original', w_or_e,
                                 city_name)

        process(dir_path, dirc_path, timestamp, command, keyword=keyword)

    else:
        raise ValueError('the argment `command` should be {\'all\' or \'add\'')
