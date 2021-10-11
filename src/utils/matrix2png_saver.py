import sys
from queue import Queue
from threading import Thread

import numpy as np
import matplotlib.pyplot as plt

res_path = '../dataset/results/'
treads_number = 10


def green2rgb(image):
    res = np.zeros((image.shape[0], image.shape[1], 3))
    image[image < 0.05] = 0
    res[:, :, 1] = image[:, :, 0]
    return res


def check_results(results):
    min_v = 1
    max_v = 0
    for image in results:
        min_v = min(min_v, image.min())
        max_v = max(max_v, image.max())
    print('min: ' + str(min_v) + ' max: ' + str(max_v))


class G2RGBTransformWorker(Thread):

    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            image, dir, i = self.queue.get()
            try:
                res = green2rgb(image)
                plt.imsave((res_path + dir + '{:04d}'.format(i) + '.png'), res)
                sys.stdout.write("\rImage %i transformed" % i)
            finally:
                self.queue.task_done()


def transform_results(results, dir):
    queue = Queue()
    for x in range(treads_number):
        worker = G2RGBTransformWorker(queue)
        worker.daemon = True
        worker.start()
    for image, i in zip(results, range(len(results))):
        queue.put((image, dir, i))
    queue.join()
