import sys
import matplotlib.pyplot as plt
import numpy as np

from queue import Queue
from threading import Thread

masks_path = '../dataset/results'
img_x, img_y = 512, 512
res_path = '../dataset/test_cases'
treads_number = 10


def green2rgb(image):
    res = np.zeros((image.shape[0], image.shape[1], 4))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i][j][0] > 0.5):
                res[i][j][1] = image[i][j]
                res[i][j][3] = 1
    return res


def green_and_red2rgb(image_g, image_r):
    res = np.zeros((image_g.shape[0], image_g.shape[1], 4))
    for i in range(image_g.shape[0]):
        for j in range(image_g.shape[1]):
            if (image_g[i][j][0] > 0.5):
                res[i][j][1] = image_g[i][j]
                res[i][j][3] = 1
            if (image_r[i][j][0] > 0.5):
                res[i][j][0] = image_r[i][j]
                res[i][j][3] = 1
    return res


def xor_images(full_img, del_img):
    res = np.zeros((full_img.shape[0], full_img.shape[1], 1))
    for i in range(img_x):
        for j in range(img_y):
            res[i][j] = max(0, (full_img[i][j] - del_img[i][j]))
    return res


def load_masks(num):
    return np.load(masks_path + '/mask_' + str(num) + '.npy')


def load_gener(num):
    return np.load(masks_path + '/gener_' + str(num) + '.npy')


# -------------------------------------------------------------

class XorTransformWorker(Thread):

    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            result, iter, full_image, deleter_image = self.queue.get()
            try:
                result[iter] = xor_images(full_image, deleter_image)
            finally:
                sys.stdout.write("\rImage %i transformed" % iter)
                self.queue.task_done()


def xor_image_sets(full, deleter):
    result = np.zeros((len(full), img_x, img_y, 1), dtype=np.float16)
    queue = Queue()
    for x in range(treads_number):
        worker = XorTransformWorker(queue)
        worker.daemon = True
        worker.start()
    for full_image, deleter_image, iter in zip(full, deleter, range(len(full))):
        queue.put((result, iter, full_image, deleter_image))
    queue.join()
    return result


class G2RGBTransformWorker(Thread):

    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            image, dir, i = self.queue.get()
            try:
                res = green2rgb(image)
                plt.imsave((res_path + dir + '{:04d}'.format(i) + '.png'), res)
                sys.stdout.write("\rImage %i transformed" % i)
            finally:
                self.queue.task_done()


def transform_results_g2rgb(results, dir):
    queue = Queue()
    for x in range(treads_number):
        worker = G2RGBTransformWorker(queue)
        worker.daemon = True
        worker.start()
    for image, i in zip(results, range(len(results))):
        queue.put((image, dir, i))
    queue.join()


class GR2RGBTransformWorker(Thread):

    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            image_g, image_r, i, dir = self.queue.get()
            try:
                res = green_and_red2rgb(image_g, image_r)
                plt.imsave((res_path + dir + '{:04d}'.format(i) + '.png'), res)
                sys.stdout.write("\rImage %i transformed" % i)
            finally:
                self.queue.task_done()


def transform_results_gr2rgb(results_g, results_r, dir):
    queue = Queue()
    for x in range(treads_number):
        worker = GR2RGBTransformWorker(queue)
        worker.daemon = True
        worker.start()
    for image_r, image_g, i in zip(results_g, results_r, range(len(results_g))):
        queue.put((image_g, image_r, i, dir))
    queue.join()


# -------------------------------------------------------------

for i in range(5, 9):
    masks = load_masks(i)
    print('\nData loaded for patient ' + str(i))
    dir_part = '/{:03d}'.format(i) + '/{:03d}'.format(i)
    print('Saving cleared data for patient ' + str(i))
    transform_results_g2rgb(masks, dir_part + '_mask_bone/')
    print('\nSaving generated data for patient ' + str(i))

