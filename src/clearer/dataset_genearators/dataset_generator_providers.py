import glob
import sys
from queue import Queue
from threading import Thread

import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from src.utils.augmentations import augment_image

dataset_path = '../dataset/train_cases'
test_dataset_path = '../dataset/test_cases'
paths_pred_masks = '/*/ORIG/*.png'
paths_answ_masks = '/*/NG/*.png'
paths_test_masks = '/*/*.png'

img_x, img_y = 512, 512
parallel_augment = 9
batch_size = 3
buffer_size = 30
treads_loader_number = 10


def rgb2green(image):
    return np.float32(np.multiply(image[:, :, 1], image[:, :, 3]))


class LoadDataWorker(Thread):

    def __init__(self, queue_in, queue_out):
        Thread.__init__(self)
        self.queue_in = queue_in
        self.queue_out = queue_out

    def run(self):
        while True:
            pred_path, img_path = self.queue_in.get()
            try:
                answer = rgb2green(plt.imread(img_path))
                predictor = np.copy(cv2.cvtColor(cv2.imread(pred_path), cv2.COLOR_RGB2GRAY)) / 255
                pred, answ = clearer_dataset_pair_augmentation(predictor, answer)
                self.queue_out.put((pred, answ))
            finally:
                self.queue_in.task_done()


def clearer_dataset_pair_generator_parallel():
    paths_answers = natsorted(glob.glob(dataset_path + paths_answ_masks))
    paths_predicts = natsorted(glob.glob(dataset_path + paths_pred_masks))
    queue_in_worker = Queue()
    queue_out_worker = Queue(maxsize=buffer_size)
    for x in range(treads_loader_number):
        worker = LoadDataWorker(queue_in_worker, queue_out_worker)
        worker.daemon = True
        worker.start()
    for path_answer, path_predictor in zip(paths_answers, paths_predicts):
        queue_in_worker.put((path_predictor, path_answer))
    for iterator in range(len(paths_answers)):
        pred, answ = queue_out_worker.get()
        queue_out_worker.task_done()
        yield pred, answ
    queue_in_worker.join()
    queue_out_worker.join()


def clearer_dataset_pair_generator():
    paths_answers = natsorted(glob.glob(dataset_path + paths_answ_masks))
    paths_predicts = natsorted(glob.glob(dataset_path + paths_pred_masks))
    for path_answer, path_predictor, i in zip(paths_answers, paths_predicts, range(len(paths_answers))):
        answer = rgb2green(plt.imread(path_answer))
        predictor = np.copy(cv2.cvtColor(cv2.imread(path_predictor), cv2.COLOR_RGB2GRAY)) / 255
        yield clearer_dataset_pair_augmentation(predictor, answer)


def transform_from_enum(enum, data):
    return data[0], data[1]


def clearer_dataset_pair_augmentation(pred, answ):
    if pred.shape[0] != img_x or pred.shape[1] != img_y:
        pred = cv2.resize(pred, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
    if answ.shape[0] != img_x or answ.shape[1] != img_y:
        answ = cv2.resize(answ, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
    pred, answ = augment_image(pred, answ)
    return pred.reshape((img_x, img_y, 1)), answ.reshape((img_x, img_y, 1))


def clearer_dataset_pair_creater():
    dataset = tf.data.Dataset.from_generator(
        clearer_dataset_pair_generator_parallel,
        output_signature=(
            tf.TensorSpec(shape=[512, 512, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[512, 512, 1], dtype=tf.float32)
        )
    )

    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size, num_parallel_calls=batch_size)
    dataset = dataset.prefetch(buffer_size=buffer_size)

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
