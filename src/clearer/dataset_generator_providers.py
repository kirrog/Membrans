import glob
import sys

import cupy as cp
import cv2
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted

dataset_path = '../dataset/train_cases'
test_dataset_path = '../dataset/hard_cases'
predict_path = '../dataset/numpys/predictors.npy'
answers_path = '../dataset/numpys/answers.npy'
tests_path = '../dataset/numpys/test.npy'
paths_pred_masks = '/*/ORIG/*.png'
paths_answ_masks = '/*/NG/*.png'
paths_test_masks = '/*/*.png'

img_x, img_y = 512, 512


def rgb2green(image):
    return np.multiply(image[:, :, 1], image[:, :, 3])


def clearer_dataset_pair():
    paths_answers = natsorted(glob.glob(dataset_path + paths_answ_masks))
    paths_predicts = natsorted(glob.glob(dataset_path + paths_pred_masks))
    for path_answer, path_predictor, i in zip(paths_answers, paths_predicts, range(len(paths_answers))):
        answer = rgb2green(plt.imread(path_answer))
        predictor = cv2.cvtColor(cv2.imread(path_predictor), cv2.COLOR_RGB2GRAY)
        predictor = np.copy(predictor) / 255
        yield predictor, answer


def clearer_dataset_answers_generator():
    paths_answers = natsorted(glob.glob(dataset_path + paths_answ_masks))
    for path_answer, i in zip(paths_answers, range(len(paths_answers))):
        answer = rgb2green(plt.imread(path_answer))
        yield answer


def clearer_dataset_predicts_generator():
    paths_predicts = natsorted(glob.glob(dataset_path + paths_pred_masks))
    for path_predictor, i in zip(paths_predicts, range(len(paths_predicts))):
        predictor = cv2.cvtColor(cv2.imread(path_predictor), cv2.COLOR_RGB2GRAY)
        yield np.copy(predictor) / 255
