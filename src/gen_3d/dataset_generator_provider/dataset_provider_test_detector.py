import _queue
import glob
import logging
from collections import defaultdict
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

paths_pred_masks = '*/G/*.png'
paths_answ_masks = '*/RG/*.png'

img_x, img_y = 512, 512
parallel_augment = 1
batch_size = 16
queue_buffer_size = 200000
buffer_size = 2000
treads_loader_number = 32
threads_prework_number = 24
step_of_x_model = 32
step_of_all_dims = 8
slice_height = 32


def red2rgb(image):
    res = np.zeros((image.shape[0], image.shape[1], 4))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i][j] > 0.5):
                res[i][j][0] = image[i][j]
                res[i][j][3] = 1
    return res


def rgb2green(image):
    return np.float32(np.multiply(image[:, :, 1], image[:, :, 3]))


def rgb2red(image):
    return np.float32(np.multiply(image[:, :, 0], image[:, :, 3]))


class LoadDataWorker(Thread):

    def __init__(self, queue_in, queue_out):
        Thread.__init__(self)
        self.queue_in = queue_in
        self.queue_out = queue_out

    def run(self):
        try:
            while True:
                pred_path = self.queue_in.get(10)
                p = Path(pred_path)
                try:
                    predictor = rgb2green(plt.imread(pred_path))
                    if predictor.shape[0] != img_x or predictor.shape[1] != img_y:
                        predictor = cv2.resize(predictor, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
                    self.queue_out.put((predictor, p.parent.parent.name, int(p.name[-8:-4])))
                finally:
                    self.queue_in.task_done()
        except _queue.Empty:
            logging.warning("Load worker end")


def apply_model(model, dataset_path: str, result_path: str):
    paths_predicts = sorted(glob.glob(dataset_path + paths_pred_masks))
    queue_in_worker = Queue()
    queue_out_worker = Queue(maxsize=buffer_size)
    for x in range(treads_loader_number):
        worker = LoadDataWorker(queue_in_worker, queue_out_worker)
        worker.daemon = True
        worker.start()
    for path_predictor in paths_predicts:
        queue_in_worker.put((path_predictor))
    result_accum = defaultdict(list)
    for iterator in range(len(paths_predicts)):
        pred, directory_num, img_num = queue_out_worker.get()
        result_accum[directory_num].append((img_num, pred))
        queue_out_worker.task_done()
    queue_in_worker.join()
    queue_out_worker.join()
    data = []
    for d in result_accum.values():
        case_len = len(d)
        image_data = np.zeros((img_x, img_y, case_len), dtype=float)
        for img_num, pred in d:
            image_data[:, :, img_num] = pred
        data.append((case_len, image_data))
    result_accum = None
    logging.warning("Loading complete!")
    case_num = 0
    for case_len, image_data in data:
        result_data = np.zeros((img_x, img_y, case_len), dtype=float)
        diver_matrix = np.zeros((img_x, img_y, case_len), dtype=float)
        p_slice_height = int(slice_height / 2)
        l = np.zeros((int(step_of_x_model * (img_y - p_slice_height - p_slice_height) / step_of_all_dims),
                      slice_height, slice_height, slice_height, 1))
        for step_z in tqdm(range(p_slice_height, case_len - p_slice_height, step_of_all_dims),
                           desc="processing images"):
            for step_x in range(p_slice_height, img_x - p_slice_height, step_of_all_dims):
                if (step_x - p_slice_height) / step_of_all_dims > 0 and (
                        step_x - p_slice_height) / step_of_all_dims % step_of_x_model == 0:
                    r = model.predict(x=l, batch_size=batch_size)
                    for step_x_ in range(step_x - step_of_x_model, step_x, step_of_all_dims):
                        for step_y_ in range(p_slice_height, img_y - p_slice_height, step_of_all_dims):
                            result_data[step_x_ - p_slice_height:step_x_ + p_slice_height,
                            step_y_ - p_slice_height:step_y_ + p_slice_height,
                            step_z - p_slice_height:step_z + p_slice_height] += np.reshape(
                                r[int((step_y_ - p_slice_height) / step_of_all_dims) + int(
                                    (step_x_ - p_slice_height) / step_of_all_dims % step_of_x_model) *
                                  int((img_y - p_slice_height - p_slice_height) / step_of_all_dims)],
                                (slice_height, slice_height,
                                 slice_height))
                            diver_matrix[step_x_ - p_slice_height:step_x_ + p_slice_height,
                            step_y_ - p_slice_height:step_y_ + p_slice_height,
                            step_z - p_slice_height:step_z + p_slice_height] += 1
                for step_y in range(p_slice_height, img_y - p_slice_height, step_of_all_dims):
                    pred_img = image_data[step_x - p_slice_height:step_x + p_slice_height,
                               step_y - p_slice_height:step_y + p_slice_height,
                               step_z - p_slice_height:step_z + p_slice_height]
                    img_slice = np.reshape(pred_img,
                                           (slice_height, slice_height, slice_height, 1))
                    l[int((step_y - p_slice_height) / step_of_all_dims) +
                      int(((step_x - p_slice_height) / step_of_all_dims) % step_of_x_model) *
                      int((img_y - p_slice_height - p_slice_height) / step_of_all_dims)] = img_slice
        p = Path(result_path) / f"{case_num:03d}"
        p.mkdir(parents=True, exist_ok=True)
        p_r = p / "res.npy"
        p_d = p / "div.npy"
        if not p_r.exists():
            np.save(str(p_r), result_data)
        if not p_d.exists():
            np.save(str(p_d), diver_matrix)
        print(result_data.max())
        print(result_data.mean())
        print(result_data.min())
        diver_matrix[diver_matrix == 0.0] = 1

        result_data[result_data < 0.0] = 0.0
        result_data /= result_data.max()

        for image_num in tqdm(range(case_len), desc="saving images"):
            plt.imsave(p / f"{image_num:03d}.png", red2rgb(result_data[:, :, image_num]))
        case_num+=1
