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

paths_pred_masks = '*/G/*.png'
paths_answ_masks = '*/R/*.png'

img_x, img_y = 512, 512
parallel_augment = 1
batch_size = 16
queue_buffer_size = 20000
buffer_size = 20000
treads_loader_number = 32
threads_prework_number = 12

step_of_search = 16
step_of_near_search = 2 
near_search_range = 8

slice_height = 32
p_slice_height = int(slice_height / 2)


treshold = 0.5
scale = 1

def rgb2green(image):
    if image.shape[2] > 3:
        if image[:,:,3].max() > 0:
            return np.float32(np.multiply(image[:, :, 1], image[:, :, 3]))
    return np.float32(image[:, :, 1])


def rgb2red(image):
    if image.shape[2] > 3:
        if image[:,:,3].max() > 0:
            return np.float32(np.multiply(image[:, :, 0], image[:, :, 3]))
    return np.float32(image[:, :, 0])


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
                            answ_img = image_data[step_x - p_slice_height:step_x + p_slice_height,
                                       step_y - p_slice_height:step_y + p_slice_height,
                                       step_z - p_slice_height:step_z + p_slice_height]
                            if pred_img.max() < treshold and answ_img.max() < treshold:
                                continue
                            if answ_img.max() > pred_img.max():
                                self.queue_out.put((self.get_views_by_coords(step_x, step_y, step_z,
                                                                        image_data, answ_data, answ_weights),1))
                            else:
                                self.queue_out.put((self.get_views_by_coords(step_x, step_y, step_z,
                                                                        image_data, answ_data, answ_weights),0))
                            length += 1
                            if answ_data[step_x, step_y, step_z] > treshold:
                                l = self.get_neighbours(step_x, step_y, step_z, answ_data)
                                length += len(l)
                                for x, y, z in l:
                                    self.queue_out.put((self.get_views_by_coords(x, y, z,
                                                                                image_data, answ_data, answ_weights),1))
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
                    if (i != 0 and j != 0 and k != 0) and tensor_[x + i, y + j, z + k] < treshold:
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
                answer = rgb2green(plt.imread(img_path))
                predictor = rgb2green(plt.imread(pred_path))

                if answer.shape[0] != img_x or answer.shape[1] != img_y:
                    answer = cv2.resize(answer, (img_x, img_y), interpolation=cv2.INTER_CUBIC)

                if predictor.shape[0] != img_x or predictor.shape[1] != img_y:
                    predictor = cv2.resize(predictor, (img_x, img_y), interpolation=cv2.INTER_CUBIC)

                self.queue_out.put((predictor, answer, p.parent.parent.name, int(p.name[-8:-4])))
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
        answ_data[answ_data < treshold] = 0.0
        answ_data[answ_data > 1] = 1.0
        answ_weights = answ_data.copy()
        answ_weights *= scale
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
    bone_counter = 0
    memb_counter = 0
    try:
        while True:
            data, type_id = queue_out_worker.get(timeout=10)
            if type_id == 0:
                bone_counter+=1
            else:
                memb_counter+=1
            img_slice, img_type, img_weight = data
            l.append((img_slice, img_type, img_weight))
    except _queue.Empty:
        pass

    logging.warning(f"Founded: all: {len(l)} bone: {bone_counter} memb: {memb_counter}")

    def generator_dataset_pair_generator_parallel():
        for i in l:
            yield i

    return generator_dataset_pair_generator_parallel


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

    return dataset
