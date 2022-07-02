import glob
import sys
from queue import Queue
from threading import Thread

import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from tqdm import tqdm

from src.utils.augmentations import augment_image

paths_pred_masks = '*/NG/*.png'
paths_answ_masks = '*/RG/*.png'
paths_pred_numbers = '/numbers.csv'

img_x, img_y = 512, 512
parallel_augment = 9
batch_size = 3
buffer_size = 30
treads_loader_number = 10


def rgb2green(image):
    return np.float32(np.multiply(image[:, :, 1], image[:, :, 3]))


def rgb2red(image):
    return np.float32(np.multiply(image[:, :, 0], image[:, :, 3]))


def read_csv(path):
    res = np.zeros((1, 512, 1), dtype=float)
    with open(path) as f:
        res_s = f.read()
        res_strs = res_s.split(',')
        for i in res_strs:
            i = int(i)
            part = int(i / 10)
            n = i % 10
            res[0, (part - 1) * 8 + n, 0] = 1
    return res


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
                predictor = rgb2green(plt.imread(pred_path))

                num = read_csv("/" + "/".join(str(pred_path).split('/')[:-2]) + paths_pred_numbers)

                if answer.shape[0] != img_x or answer.shape[1] != img_y:
                    answer = cv2.resize(answer, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
                if predictor.shape[0] != img_x or predictor.shape[1] != img_y:
                    predictor = cv2.resize(predictor, (img_x, img_y), interpolation=cv2.INTER_CUBIC)

                pred, answ = augment_image(predictor, answer)
                pred, answ = generator_dataset_pair_augmentation(pred, answ)

                pred = np.concatenate((pred, num), axis=0)

                self.queue_out.put((pred, answ))
            finally:
                self.queue_in.task_done()


def generator_dataset_pair_generator_parallel_getter(dataset_path):
    def generator_dataset_pair_generator_parallel():
        paths_answers = natsorted(glob.glob(dataset_path + paths_answ_masks))
        paths_predicts = natsorted(glob.glob(dataset_path + paths_pred_masks))
        queue_in_worker = Queue()
        queue_out_worker = Queue(maxsize=buffer_size)
        for x in range(treads_loader_number):
            worker = LoadDataWorker(queue_in_worker, queue_out_worker)
            worker.daemon = True
            worker.start()
        for path_answer, path_predictor in tqdm(zip(paths_answers, paths_predicts)):
            queue_in_worker.put((path_predictor, path_answer))
        for iterator in range(len(paths_answers)):
            pred, answ = queue_out_worker.get()
            queue_out_worker.task_done()
            yield pred, answ
        queue_in_worker.join()
        queue_out_worker.join()

    return generator_dataset_pair_generator_parallel


def transform_from_enum(enum, data):
    return data[0], data[1]


def generator_dataset_pair_augmentation(pred, answ):
    if pred.shape[0] != img_x or pred.shape[1] != img_y:
        pred = cv2.resize(pred, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
    if answ.shape[0] != img_x or answ.shape[1] != img_y:
        answ = cv2.resize(answ, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
    pred, answ = augment_image(pred, answ)
    return pred.reshape((img_x, img_y, 1)), answ.reshape((img_x, img_y, 1))


def generator_dataset_pair_creater(data_path):
    dataset = tf.data.Dataset.from_generator(
        generator_dataset_pair_generator_parallel_getter(data_path),
        output_signature=(
            tf.TensorSpec(shape=[513, 512, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[512, 512, 1], dtype=tf.float32),
            # tf.TensorSpec(shape=[32, 1], dtype=tf.float32)
        )
    )

    # dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size, num_parallel_calls=batch_size)
    dataset = dataset.prefetch(buffer_size=buffer_size)

    dataset = dataset.enumerate() \
        .map(transform_from_enum, num_parallel_calls=parallel_augment)

    return dataset
