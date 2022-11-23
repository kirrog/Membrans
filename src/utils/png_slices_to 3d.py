import glob
import time

import cv2
import numpy as np
from natsort import natsorted


def find_png_files(directory):
    png_files = []
    files = natsorted(glob.glob(directory + '/*.png'))
    for file in files:
        png_files.append(file)
    return png_files


def convert_png(directory, png_shape=()):
    png_files = find_png_files(directory)

    if not png_shape:
        ref_data = cv2.imread(png_files[0], 0).shape  # color, gray, unchanged ?
        out_shape = (len(png_files), int(ref_data[0]), int(ref_data[1]))  # [0] and [1] for gray img
    else:
        out_shape = (len(png_files), png_shape[0], png_shape[1])  # [0] and [1] for gray img

    png_np_array = np.zeros(out_shape, dtype=int)

    for png_file in png_files:
        png_img = cv2.imread(png_file, 0)

        if png_img.shape[0] < out_shape[1] or png_img.shape[1] < out_shape[2]:
            png_img = cv2.resize(png_img, (out_shape[2], out_shape[1]), interpolation=cv2.INTER_CUBIC)
        elif png_img.shape[0] > out_shape[1] or png_img.shape[1] > out_shape[2]:
            png_img = cv2.resize(png_img, (out_shape[2], out_shape[1]), interpolation=cv2.INTER_AREA)

        png_np_array[png_files.index(png_file), :, :] = png_img

    return png_np_array


if __name__ == '__main__':
    test_dir = ''
    start = time.time()
    arr = convert_png(test_dir)  # con specify png_shape otherwise shape of the first file
    print(arr.shape)
    end = time.time()
    print('Execution time: {}'.format(end-start))  # 0.039 sec for 4 png file
