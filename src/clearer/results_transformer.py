import sys

import numpy as np
import matplotlib.pyplot as plt

res_path = './dataset/results/'


def green2rgb(image):
    res = np.zeros((image.shape[0], image.shape[1], 4))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i][j][0] > 0.5):
                res[i][j][1] = image[i][j]
                res[i][j][3] = 1
    return res


def check_results(results):
    min_v = 1
    max_v = 0
    for image in results:
        min_v = min(image.min(), min_v)
        max_v = max(image.max(), max_v)
    print('min: ' + str(min_v) + ' max: ' + str(max_v))


def transform_results(results, dir, start):
    for image, i in zip(results, range(len(results))):
        res = green2rgb(image)
        plt.imsave((res_path + dir + '{:04d}'.format((start + i)) + '.png'), res)
        sys.stdout.write("\rImage %i transformed" % i)
