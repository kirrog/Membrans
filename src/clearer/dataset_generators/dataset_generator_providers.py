import glob
import sys
from queue import Queue
from threading import Thread

import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
import albumentations as albu


def aug_transforms():
    return [
        albu.VerticalFlip(),
        albu.HorizontalFlip(),
        albu.Rotate(limit=180, interpolation=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_WRAP, always_apply=False,
                    p=0.6),
        albu.ElasticTransform(alpha=10, sigma=50, alpha_affine=28,
                              interpolation=cv2.INTER_LANCZOS4, border_mode=cv2.BORDER_WRAP,
                              always_apply=False, approximate=False, p=0.6),
        albu.GridDistortion(num_steps=20, distort_limit=0.2, interpolation=cv2.INTER_LANCZOS4,
                            border_mode=cv2.BORDER_WRAP,
                            always_apply=False, p=0.5)
    ]


transforms = albu.Compose(aug_transforms())


def augment_image(image, mask):
    res = transforms(image=image, mask=mask)
    return res["image"], res["mask"]


paths_pred_masks = '*/ORIG/*.png'
paths_answ_masks = '*/NG/*.png'

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

                if answer.shape[0] != img_x or answer.shape[1] != img_y:
                    answer = cv2.resize(answer, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
                if predictor.shape[0] != img_x or predictor.shape[1] != img_y:
                    predictor = cv2.resize(predictor, (img_x, img_y), interpolation=cv2.INTER_CUBIC)

                pred, answ = augment_image(predictor, answer)
                pred, answ = clearer_dataset_pair_augmentation(pred, answ)
                self.queue_out.put((pred, answ))
            finally:
                self.queue_in.task_done()


def clearer_dataset_pair_generator_parallel_getter(dataset_path):
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

    return clearer_dataset_pair_generator_parallel


def transform_from_enum(enum, data):
    return data[0], data[1]


def clearer_dataset_pair_augmentation(pred, answ):
    if pred.shape[0] != img_x or pred.shape[1] != img_y:
        pred = cv2.resize(pred, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
    if answ.shape[0] != img_x or answ.shape[1] != img_y:
        answ = cv2.resize(answ, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
    pred, answ = augment_image(pred, answ)
    return pred.reshape((img_x, img_y, 1)), answ.reshape((img_x, img_y, 1))


def clearer_dataset_pair_creater(data_path):
    dataset = tf.data.Dataset.from_generator(
        clearer_dataset_pair_generator_parallel_getter(data_path),
        output_signature=(
            tf.TensorSpec(shape=[512, 512, 1], dtype=tf.float32),
            tf.TensorSpec(shape=[512, 512, 1], dtype=tf.float32)
        )
    )

    # dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size, num_parallel_calls=batch_size)
    dataset = dataset.prefetch(buffer_size=buffer_size)

    dataset = dataset.enumerate() \
        .map(transform_from_enum, num_parallel_calls=parallel_augment)

    return dataset
