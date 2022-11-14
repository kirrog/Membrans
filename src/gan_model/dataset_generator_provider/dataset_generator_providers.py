import glob
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from src.utils.augmentations import augment_image

paths_pred_masks = '*/NG/*.png'
paths_answ_masks = '*/RG/*.png'

img_x, img_y = 512, 512
parallel_augment = 9
batch_size = 3
buffer_size = 30
treads_loader_number = 10


def rgb2green(image):
    return np.float32(np.multiply(image[:, :, 1], image[:, :, 3]))


def rgb2red(image):
    return np.float32(np.multiply(image[:, :, 0], image[:, :, 3]))


class LoadDataWorker(Thread):

    def __init__(self, queue_in, queue_out, augment):
        Thread.__init__(self)
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.augment = augment

    def run(self):
        while True:
            pred_path, img_path = self.queue_in.get()
            try:
                answer = rgb2red(plt.imread(img_path))
                predictor = np.copy(cv2.cvtColor(cv2.imread(pred_path), cv2.COLOR_RGB2GRAY)) / 255
                pred, answ = generator_dataset_pair_augmentation(predictor, answer, self.augment)
                self.queue_out.put((pred, answ))
            finally:
                self.queue_in.task_done()


def generator_dataset_pair_generator_parallel_getter(dataset_path, augment):
    def generator_dataset_pair_generator_parallel():
        paths_answers = sorted(glob.glob(dataset_path + paths_answ_masks))
        paths_predicts = sorted(glob.glob(dataset_path + paths_pred_masks))
        queue_in_worker = Queue()
        queue_out_worker = Queue(maxsize=buffer_size)
        for x in range(treads_loader_number):
            worker = LoadDataWorker(queue_in_worker, queue_out_worker, augment)
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

    return generator_dataset_pair_generator_parallel


def generator_dataset_pair_augmentation(pred, answ, augment):
    if augment:
        pred, answ = augment_image(pred, answ)
    if pred.shape[0] != img_x or pred.shape[1] != img_y:
        pred = cv2.resize(pred, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
    if answ.shape[0] != img_x or answ.shape[1] != img_y:
        answ = cv2.resize(answ, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
    return pred.reshape((img_x, img_y, 1)), answ.reshape((img_x, img_y, 1))


def generator_dataset_pair_creater(data_path, augment=True):
    dataset = tf.data.Dataset.from_generator(
        generator_dataset_pair_generator_parallel_getter(data_path, augment),
        output_signature=(
            tf.TensorSpec(shape=[512, 512, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[512, 512, 1], dtype=tf.float32)
        )
    )

    dataset = dataset.batch(batch_size, num_parallel_calls=batch_size)
    dataset = dataset.prefetch(buffer_size=buffer_size)

    return dataset
