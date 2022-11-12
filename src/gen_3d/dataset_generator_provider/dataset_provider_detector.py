import _queue
import glob
import logging
from collections import defaultdict
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

paths_pred_masks = '*/NG/*.png'
paths_answ_masks = '*/RG/*.png'

img_x, img_y = 512, 512
parallel_augment = 1
batch_size = 1
queue_buffer_size = 20000
buffer_size = 20000
treads_loader_number = 32
threads_prework_number = 12

step_of_search = 16
step_of_near_search = 1
near_search_range = 8

slice_height = 32
p_slice_height = int(slice_height / 2)


def rgb2green(image):
    return np.float32(np.multiply(image[:, :, 1], image[:, :, 3]))


def rgb2red(image):
    return np.float32(np.multiply(image[:, :, 0], image[:, :, 3]))


class FounderWorker(Thread):
    def __init__(self, queue_in, queue_out):
        Thread.__init__(self)
        self.queue_in = queue_in
        self.queue_out = queue_out

    def run(self):
        length = 0
        try:
            while True:
                case_len, image_data, answ_data, answ_weights = self.queue_in.get(10)
                for step_z in range(p_slice_height, case_len - p_slice_height, step_of_search):
                    for step_x in range(p_slice_height, img_x - p_slice_height, step_of_search):
                        for step_y in range(p_slice_height, img_y - p_slice_height, step_of_search):
                            pred_img = image_data[step_x - p_slice_height:step_x + p_slice_height,
                                       step_y - p_slice_height:step_y + p_slice_height,
                                       step_z - p_slice_height:step_z + p_slice_height]
                            if pred_img.max() < 0.5:
                                continue
                            self.queue_out.put(self.get_views_by_coords(step_x, step_y, step_z,
                                                                        image_data, answ_data, answ_weights))
                            length += 1
                            if answ_data[step_x, step_y, step_z] > 0.5:
                                l = self.get_neighbours(step_x, step_y, step_z, answ_data)
                                length += len(l)
                                for x, y, z in l:
                                    self.queue_out.put(self.get_views_by_coords(x, y, z,
                                                                                image_data, answ_data, answ_weights))

        except _queue.Empty:
            logging.warning(f"End founding {length}")

    def get_neighbours(self, x, y, z, tensor_):
        l = []
        x_shape = tensor_.shape[0]
        y_shape = tensor_.shape[1]
        z_shape = tensor_.shape[2]
        for i in range(-near_search_range, near_search_range + 1, step_of_near_search):
            for j in range(-near_search_range, near_search_range + 1, step_of_near_search):
                for k in range(-near_search_range, near_search_range + 1, step_of_near_search):
                    if (i != 0 and j != 0 and k != 0) and tensor_[x + i, y + j, z + k] < 0.5:
                        if (p_slice_height < x + i < (x_shape - p_slice_height)) and (
                                p_slice_height < y + j < (y_shape - p_slice_height)) and (
                                p_slice_height < z + k < (z_shape - p_slice_height)):
                            l.append((x + i, y + j, z + k))
        return l

    def get_views_by_coords(self, x: int, y: int, z: int, image_data, answ_data, answ_weights):
        img_img = answ_data[x - p_slice_height:x + p_slice_height,
                  y - p_slice_height:y + p_slice_height,
                  z - p_slice_height:z + p_slice_height]
        img_type = np.reshape(img_img, (slice_height, slice_height, slice_height, 1))

        pred_img = image_data[x - p_slice_height:x + p_slice_height,
                   y - p_slice_height:y + p_slice_height,
                   z - p_slice_height:z + p_slice_height]
        img_slice = np.reshape(pred_img, (slice_height, slice_height, slice_height, 1))

        answ_img = answ_weights[x - p_slice_height:x + p_slice_height,
                   y - p_slice_height:y + p_slice_height,
                   z - p_slice_height:z + p_slice_height]
        answ_wights_slice = np.reshape(answ_img, (slice_height, slice_height, slice_height, 1))

        return img_slice, img_type, answ_wights_slice


class LoadDataWorker(Thread):

    def __init__(self, queue_in, queue_out):
        Thread.__init__(self)
        self.queue_in = queue_in
        self.queue_out = queue_out

    def run(self):

        while True:
            pred_path, img_path = self.queue_in.get(10)
            p = Path(pred_path)
            try:
                answer = rgb2red(plt.imread(img_path))
                predictor = rgb2green(plt.imread(pred_path))

                if answer.shape[0] != img_x or answer.shape[1] != img_y:
                    answer = cv2.resize(answer, (img_x, img_y), interpolation=cv2.INTER_CUBIC)

                if predictor.shape[0] != img_x or predictor.shape[1] != img_y:
                    predictor = cv2.resize(predictor, (img_x, img_y), interpolation=cv2.INTER_CUBIC)

                self.queue_out.put((predictor, answer, int(p.parent.parent.name), int(p.name[-8:-4])))
            finally:
                self.queue_in.task_done()


def generator_dataset_pair_generator_parallel_getter(dataset_path):
    paths_answers = sorted(glob.glob(dataset_path + paths_answ_masks))
    paths_predicts = sorted(glob.glob(dataset_path + paths_pred_masks))
    queue_in_worker = Queue()
    queue_out_worker = Queue(maxsize=buffer_size)
    for x in range(treads_loader_number):
        worker = LoadDataWorker(queue_in_worker, queue_out_worker)
        worker.daemon = True
        worker.start()
    for path_answer, path_predictor in zip(paths_answers, paths_predicts):
        queue_in_worker.put((path_predictor, path_answer))
    result_accum = defaultdict(list)
    for iterator in range(len(paths_answers)):
        pred, answ, directory_num, img_num = queue_out_worker.get()
        result_accum[directory_num].append((img_num, pred, answ))
        queue_out_worker.task_done()
    queue_in_worker.join()
    queue_out_worker.join()
    data = []
    for d in result_accum.values():
        case_len = len(d)
        image_data = np.zeros((img_x, img_y, case_len), dtype=float)
        answ_data = np.zeros((img_x, img_y, case_len), dtype=float)
        for img_num, pred, answ in d:
            image_data[:, :, img_num] = pred
            answ_data[:, :, img_num] = answ
        answ_data[answ_data < 0.5] = 0.0
        answ_weights = answ_data.copy()
        answ_weights *= 8
        answ_weights += 1
        data.append((case_len, image_data, answ_data, answ_weights))
    result_accum = None
    logging.warning("Loading complete!")
    l = []

    queue_in_worker = Queue()
    queue_out_worker = Queue()
    for x in range(threads_prework_number):
        worker = FounderWorker(queue_in_worker, queue_out_worker)
        worker.daemon = True
        worker.start()
    for case_len, image_data, answ_data, answ_weights in data:
        queue_in_worker.put((case_len, image_data, answ_data, answ_weights))
    try:
        while True:
            img_slice, img_type, img_weight = queue_out_worker.get(timeout=10)
            l.append((img_slice, img_type, img_weight))
    except _queue.Empty:
        pass

    logging.warning(f"Founded {len(l)}")

    def generator_dataset_pair_generator_parallel():
        for i in l:
            yield i

    return generator_dataset_pair_generator_parallel


def transform_from_enum(enum, data):
    return data[0], data[1], data[2]


def generator_dataset_pair_creater(data_path):
    dataset = tf.data.Dataset.from_generator(
        generator_dataset_pair_generator_parallel_getter(data_path),
        output_signature=(
            tf.TensorSpec(shape=[slice_height, slice_height, slice_height, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[slice_height, slice_height, slice_height, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[slice_height, slice_height, slice_height, 1], dtype=tf.float32)
        )
    )

    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size, num_parallel_calls=batch_size)
    dataset = dataset.prefetch(buffer_size=buffer_size)

    dataset = dataset.enumerate().map(transform_from_enum)

    return dataset
