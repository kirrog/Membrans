import glob
import sys

import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted

dataset_path = '../dataset/train_cases'
test_dataset_path = '../dataset/hard_cases'
paths_pred_masks = '/*/NG/*.png'
paths_answ_masks = '/*/R/*.png'

img_x, img_y = 512, 512
parallel_augment = 3
batch_size = 9
autotune = 9
buffer_size = 21

def rgb2red(image):
    return np.float32(np.multiply(image[:, :, 0], image[:, :, 3]))

def rgb2green(image):
    return np.float32(np.multiply(image[:, :, 1], image[:, :, 3]))


def generator_dataset_pair_generator():
    paths_answers = natsorted(glob.glob(dataset_path + paths_answ_masks))
    paths_predicts = natsorted(glob.glob(dataset_path + paths_pred_masks))
    # print("Size of dataset is: " + str(len(paths_predicts)))
    for path_answer, path_predictor, i in zip(paths_answers, paths_predicts, range(len(paths_answers))):
        answer = rgb2red(plt.imread(path_answer))
        predictor = rgb2green(plt.imread(path_predictor))
        yield generator_dataset_pair_augmentation(predictor, answer)


def transform_from_enum(enum, data):
    return data[0], data[1]


def generator_dataset_pair_augmentation(pred, answ):
    if pred.shape[0] != img_x or pred.shape[1] != img_y:
        pred = cv2.resize(pred, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
    if answ.shape[0] != img_x or answ.shape[1] != img_y:
        answ = cv2.resize(answ, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
    return pred.reshape((img_x, img_y, 1)), answ.reshape((img_x, img_y, 1))


def generator_dataset_pair_creater():
    dataset = tf.data.Dataset.from_generator(
        generator_dataset_pair_generator,
        output_signature=(
            tf.TensorSpec(shape=[512, 512, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[512, 512, 1], dtype=tf.float32)
        )
    )

    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=autotune)

    def is_test(x, _):
        return x % 10 == 0

    def is_train(x, y):
        return not is_test(x, y)

    test_dataset = dataset.enumerate() \
        .filter(is_test) \
        .map(transform_from_enum, num_parallel_calls=parallel_augment)

    train_dataset = dataset.enumerate() \
        .filter(is_train) \
        .map(transform_from_enum, num_parallel_calls=parallel_augment)

    return train_dataset, test_dataset


def generator_dataset_answers_generator():
    paths_answers = natsorted(glob.glob(dataset_path + paths_answ_masks))
    for path_answer, i in zip(paths_answers, range(len(paths_answers))):
        answer = rgb2green(plt.imread(path_answer))
        yield answer


def generator_dataset_predicts_generator():
    paths_predicts = natsorted(glob.glob(dataset_path + paths_pred_masks))
    for path_predictor, i in zip(paths_predicts, range(len(paths_predicts))):
        predictor = cv2.cvtColor(cv2.imread(path_predictor), cv2.COLOR_RGB2GRAY)
        yield np.copy(predictor) / 255
