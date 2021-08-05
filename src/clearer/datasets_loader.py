import glob
import os
import sys
import cv2
from natsort import natsorted
import matplotlib.pyplot as plt
import numpy as np

dataset_path = '\\..\\dataset\\easy_cases'
img_x, img_y = 512, 512


def rgb2green(image):
    res = np.zeros((image.shape[0],image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if(image[i][j][1] == image[i][j][3]):
                res[i][j] = image[i][j][1]
            else:
                res[i][j] = 0
    return res

def load_clearer_dataset():
    cwd = os.getcwd()
    dataFold = cwd + dataset_path
    paths_predicts = natsorted(glob.glob(dataFold + '\\*\\*_image\\*.png'))
    paths_answers = natsorted(glob.glob(dataFold + '\\*\\*_mask_bone\\*.png'))

    set_predictors = np.zeros((len(paths_predicts), img_x, img_y, 1), dtype=np.float32)
    set_answers = np.zeros((len(paths_answers), img_x, img_y, 1), dtype=np.uint8)

    assert set_predictors.shape[0] == set_answers.shape[0], \
        'Error: the amount of images doesn''t correspond masks ones, {0:d} vs {1:d}' \
            .format(set_predictors.shape[0], set_answers.shape[0])

    for path_predictor, path_answer, i in zip(paths_predicts, paths_answers, range(len(paths_predicts))):
        predictor = cv2.cvtColor(cv2.imread(path_predictor),cv2.COLOR_RGB2GRAY)
        answer = rgb2green(plt.imread(path_answer))

        if len(predictor.shape) > 2:
            set_predictors[i, ..., 0] = np.copy(predictor[..., 0])
        else:
            set_predictors[i, ..., 0] = np.copy(predictor)

        set_predictors[i, ..., 0] /= 255
        set_answers[i, ..., 0] = answer[..., 0]
        sys.stdout.write("\rImage %i loaded" % i)

    return set_predictors, set_answers

def add_augmentated_image(predict, answ, collection_predict, collection_answer):
    print('add_augmentated_image')


def augmentate_image(image):
    print('augmentate_image')