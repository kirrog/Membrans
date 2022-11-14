import glob
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

from src.utils.config_loader import batch_size
from src.utils.config_loader import model_path
from src.utils.config_loader import one_test_data_dir
from src.utils.config_loader import output_data_dir

img_x, img_y = 512, 512
model = keras.models.load_model(model_path)
model.summary()


def rgb2green(image):
    return np.float32(np.multiply(image[:, :, 1], image[:, :, 2]))


def read_csv(path):
    res = np.zeros((1, 512), dtype=float)
    with open(path) as f:
        res_s = f.read()
        res_strs = res_s.split(',')
        for i in res_strs:
            i = int(i)
            part = int(i / 10)
            n = i % 10
            res[0, (part - 1) * 8 + n] = 1
    return res


def add_numbers(pred, numbers_path):
    numbers_data = read_csv(numbers_path)
    data = np.zeros((pred.shape[0], 1, 512))
    for i in range(pred.shape[0]):
        data[i] = numbers_data
    res = np.concatenate((pred, data), axis=1)
    return res


def read_from_png(directory):
    paths_predicts = sorted(glob.glob(directory + 'ORIG/*.png'))
    print(len(paths_predicts))
    pred = np.zeros([len(paths_predicts), 512, 512])
    for file, iter in zip(paths_predicts, range(len(paths_predicts))):
        image_2d = rgb2green(plt.imread(file))
        image_2d = (np.maximum(image_2d, 0) / image_2d.max())
        image_2d = cv2.resize(image_2d, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
        pred[iter, :, :] = image_2d
        sys.stdout.write("\rImage %i loaded" % iter)
    print(pred.shape[0])
    return pred


data = read_from_png(one_test_data_dir)
print(data.shape)
# data = add_numbers(data, one_test_data_dir + "/numbers.csv")
results = model.predict(x=data, batch_size=batch_size)


def from_npy_to_png(directory, data):
    Path(directory).mkdir(parents=True, exist_ok=True)
    for iter in range(data.shape[0]):
        d = np.zeros((data.shape[1], data.shape[2], 3))
        d[:, :, 0] = data[iter, :, :, 0]
        plt.imsave(str(directory) + f"/{iter:04d}.png", d)
        sys.stdout.write("\rImage %i written" % iter)


from_npy_to_png(output_data_dir, results)
