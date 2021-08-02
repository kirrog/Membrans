import glob
import os

import cv2
from natsort import natsorted
import numpy as np

dataset_path = '\\..\\dataset\\easy_cases'
img_x, img_y = 512, 512


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
        predictor = cv2.imread(path_predictor)
        answer = cv2.imread(path_answer)

        if len(predictor.shape) > 2:
            set_predictors[i, ..., 0] = np.copy(predictor[..., 0])
        else:
            set_predictors[i, ..., 0] = np.copy(predictor)

        set_predictors[i, ..., 0] /= 255
        set_answers[i, ..., 0] = answer[..., 0]

    return set_predictors, set_answers
