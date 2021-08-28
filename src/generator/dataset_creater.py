

import glob
import os
import sys
import cv2
from natsort import natsorted
import matplotlib.pyplot as plt
import numpy as np

# dataset_path = '/content/drive/MyDrive/Membrans/dataset/easy_cases'
# test_dataset_path = '/content/drive/MyDrive/Membrans/dataset/hard_cases'
# predict_path = '/content/drive/MyDrive/Membrans/dataset/numpys/generator/predictors.npy'
# answers_path = '/content/drive/MyDrive/Membrans/dataset/numpys/generator/answers.npy'
# tests_path = '/content/drive/MyDrive/Membrans/dataset/numpys/generator/test.npy'
# paths_pred_masks = '/*/*_image/*.png'
# paths_answ_masks = '/*/*_mask_bone/*.png'
# paths_test_masks = '/*/*.png'
from src.generator.datasets_loader import save_generator_dataset_predicts, save_generator_dataset_answers, \
    save_generator_dataset_tests

dataset_path = '\\..\\dataset\\easy_cases'
test_dataset_path = '\\..\\dataset\\hard_cases'
predict_path = '..\\dataset\\numpys\\generator\\predictors.npy'
answers_path = '..\\dataset\\numpys\\generator\\answers.npy'
tests_path = '..\\dataset\\numpys\\generator\\test.npy'
paths_pred_masks = '\\*\\*_mask_bone\\*.png'
paths_answ_masks = '\\*\\*_mask_bone_membr_onecol\\*.png'
paths_test_masks = '\\*\\*_mask_bone\\*.png'

img_x, img_y = 512, 512


def rgb2green(image):
    res = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float16)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][3] > 0:
                res[i][j][0] = image[i][j][1]
    return res


def rgb2red(image):
    res = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float16)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][3] > 0:
                res[i][j][0] = image[i][j][0]
    return res


def load_generator_dataset_from_images():
    data_fold = os.getcwd() + dataset_path
    paths_predicts = natsorted(glob.glob(data_fold + paths_pred_masks))
    paths_answers = natsorted(glob.glob(data_fold + paths_answ_masks))

    set_predictors = np.zeros((len(paths_predicts), img_x, img_y, 1), dtype=np.float16)
    set_answers = np.zeros((len(paths_answers), img_x, img_y, 1), dtype=np.float16)

    assert set_predictors.shape[0] == set_answers.shape[0], \
        'Error: the amount of images doesn''t correspond masks ones, {0:d} vs {1:d}' \
            .format(set_predictors.shape[0], set_answers.shape[0])

    for path_predictor, path_answer, i in zip(paths_predicts, paths_answers, range(len(paths_predicts))):
        predictor = rgb2green(plt.imread(path_predictor))
        answer = rgb2red(plt.imread(path_answer))

        set_predictors[i] = predictor
        set_answers[i] = answer
        sys.stdout.write("\rImage %i loaded" % i)

    return set_predictors, set_answers


def load_generator_predicts_from_images():
    data_fold = os.getcwd() + dataset_path
    paths_predicts = natsorted(glob.glob(data_fold + paths_pred_masks))

    set_predictors = np.zeros((len(paths_predicts), img_x, img_y, 1), dtype=np.float16)

    for path_predictor, i in zip(paths_predicts, range(len(paths_predicts))):
        predictor = rgb2green(plt.imread(path_predictor))
        set_predictors[i] = predictor
        sys.stdout.write("\rImage %i loaded" % i)
    return set_predictors


def load_generator_answers_from_images():
    data_fold = os.getcwd() + dataset_path
    paths_answers = natsorted(glob.glob(data_fold + paths_answ_masks))

    set_answers = np.zeros((len(paths_answers), img_x, img_y, 1), dtype=np.float16)

    for path_answer, i in zip(paths_answers, range(len(paths_answers))):
        answer = rgb2red(plt.imread(path_answer))
        set_answers[i] = answer
        sys.stdout.write("\rImage %i loaded" % i)
    return set_answers


def make_generator_dataset():
    pred, answ = load_generator_dataset_from_images()
    save_generator_dataset_predicts(pred)
    save_generator_dataset_answers(answ)


def load_generator_test_dataset_from_images():
    cwd = os.getcwd()
    dataFold = cwd + test_dataset_path
    paths_predicts = natsorted(glob.glob(dataFold + paths_test_masks))

    set_predictors = np.zeros((len(paths_predicts), img_x, img_y, 1), dtype=np.float16)

    for path_predictor, i in zip(paths_predicts, range(len(paths_predicts))):
        predictor = cv2.cvtColor(cv2.imread(path_predictor), cv2.COLOR_RGB2GRAY)
        set_predictors[i, ..., 0] = np.copy(predictor) / 255
        sys.stdout.write("\rImage %i loaded" % i)

    return set_predictors


def make_generator_test_dataset():
    pred = load_generator_test_dataset_from_images()
    save_generator_dataset_tests(pred)

