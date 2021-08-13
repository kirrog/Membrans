import glob
import os
import sys
import cv2
from natsort import natsorted
import matplotlib.pyplot as plt
import numpy as np

# dataset_path = '/content/drive/MyDrive/Membrans/dataset/easy_cases'
# test_dataset_path = '/content/drive/MyDrive/Membrans/dataset/hard_cases'
# predict_path = '/content/drive/MyDrive/Membrans/dataset/numpys/predictors.npy'
# answers_path = '/content/drive/MyDrive/Membrans/dataset/numpys/answers.npy'
# tests_path = '/content/drive/MyDrive/Membrans/dataset/numpys/test.npy'
# paths_pred_masks = '/*/*_image/*.png'
# paths_answ_masks = '/*/*_mask_bone/*.png'
# paths_test_masks = '/*/*.png'

dataset_path = '\\..\\dataset\\easy_cases'
test_dataset_path = '\\..\\dataset\\hard_cases'
predict_path = '..\\dataset\\numpys\\predictors.npy'
answers_path = '..\\dataset\\numpys\\answers.npy'
tests_path = '..\\dataset\\numpys\\test.npy'
paths_pred_masks = '\\*\\*_image\\*.png'
paths_answ_masks = '\\*\\*_mask_bone\\*.png'
paths_test_masks = '\\*\\*.png'

img_x, img_y = 512, 512


def rgb2green(image):
    res = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float16)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][3] > 0:
                res[i][j][0] = image[i][j][1]
    return res


def save_clearer_dataset_tests(numpy_table):
    save_clearer_dataset_to_numpy_table(numpy_table, tests_path)


def save_clearer_dataset_answers(numpy_table):
    save_clearer_dataset_to_numpy_table(numpy_table, answers_path)


def save_clearer_dataset_predicts(numpy_table):
    save_clearer_dataset_to_numpy_table(numpy_table, predict_path)


def save_clearer_dataset_to_numpy_table(numpy_table, file_path):
    np.save(file_path, numpy_table)


def load_clearer_dataset_tests():
    return load_clearer_dataset_to_numpy_table(tests_path)


def load_clearer_dataset_answers():
    return load_clearer_dataset_to_numpy_table(answers_path)


def load_clearer_dataset_predicts():
    return load_clearer_dataset_to_numpy_table(predict_path)


def load_clearer_dataset_to_numpy_table(file_path):
    return np.load(file_path)


def load_clearer_dataset_from_images():
    data_fold = os.getcwd() + dataset_path
    paths_predicts = natsorted(glob.glob(data_fold + paths_pred_masks))
    paths_answers = natsorted(glob.glob(data_fold + paths_answ_masks))

    set_predictors = np.zeros((len(paths_predicts), img_x, img_y, 1), dtype=np.float16)
    set_answers = np.zeros((len(paths_answers), img_x, img_y, 1), dtype=np.float16)

    assert set_predictors.shape[0] == set_answers.shape[0], \
        'Error: the amount of images doesn''t correspond masks ones, {0:d} vs {1:d}' \
            .format(set_predictors.shape[0], set_answers.shape[0])

    for path_predictor, path_answer, i in zip(paths_predicts, paths_answers, range(len(paths_predicts))):
        predictor = cv2.cvtColor(cv2.imread(path_predictor), cv2.COLOR_RGB2GRAY)
        answer = rgb2green(plt.imread(path_answer))

        set_predictors[i] = np.copy(predictor) / 255
        set_answers[i] = answer
        sys.stdout.write("\rImage %i loaded" % i)

    return set_predictors, set_answers


def load_clearer_predicts_from_images():
    data_fold = os.getcwd() + dataset_path
    paths_predicts = natsorted(glob.glob(data_fold + paths_pred_masks))

    set_predictors = np.zeros((len(paths_predicts), img_x, img_y, 1), dtype=np.float16)

    for path_predictor, i in zip(paths_predicts, range(len(paths_predicts))):
        predictor = cv2.cvtColor(cv2.imread(path_predictor), cv2.COLOR_RGB2GRAY)
        set_predictors[i] = np.copy(predictor) / 255
        sys.stdout.write("\rImage %i loaded" % i)
    return set_predictors


def load_clearer_answers_from_images():
    data_fold = os.getcwd() + dataset_path
    paths_answers = natsorted(glob.glob(data_fold + paths_answ_masks))

    set_answers = np.zeros((len(paths_answers), img_x, img_y, 1), dtype=np.float16)

    for path_answer, i in zip(paths_answers, range(len(paths_answers))):
        answer = rgb2green(plt.imread(path_answer))
        set_answers[i] = answer
        sys.stdout.write("\rImage %i loaded" % i)
    return set_answers


def make_clearer_dataset():
    pred, answ = load_clearer_dataset_from_images()
    save_clearer_dataset_predicts(pred)
    save_clearer_dataset_answers(answ)


def load_test_dataset_from_images():
    cwd = os.getcwd()
    dataFold = cwd + test_dataset_path
    paths_predicts = natsorted(glob.glob(dataFold + paths_test_masks))

    set_predictors = np.zeros((len(paths_predicts), img_x, img_y, 1), dtype=np.float16)

    for path_predictor, i in zip(paths_predicts, range(len(paths_predicts))):
        predictor = cv2.cvtColor(cv2.imread(path_predictor), cv2.COLOR_RGB2GRAY)
        predictor = cv2.resize(predictor, (512, 512), interpolation=cv2.INTER_CUBIC)
        set_predictors[i, ..., 0] = np.copy(predictor) / 255
        sys.stdout.write("\rImage %i loaded" % i)

    return set_predictors


def make_test_dataset():
    pred = load_test_dataset_from_images()
    save_clearer_dataset_tests(pred)


def rgb2green_test(image):
    res = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float16)
    num = 0.0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][3] > 0:
                res[i][j][0] = image[i][j][1]
                num += res[i][j][0]
            else:
                res[i][j][0] = 0
    return res, num
