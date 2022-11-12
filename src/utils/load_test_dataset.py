import glob
import sys
import cv2
from natsort import natsorted
import numpy as np

dataset_path = '../dataset/hard_cases'
img_x, img_y = 512, 512

def load_clearer_teset_set():
    dataFold = dataset_path
    paths_predicts = natsorted(glob.glob(dataFold + '/*/*.png'))

    set_predictors = np.zeros((len(paths_predicts), img_x, img_y, 1), dtype=np.float32)

    for path_predictor, i in zip(paths_predicts, range(len(paths_predicts))):
        predictor = cv2.cvtColor(cv2.imread(path_predictor), cv2.COLOR_RGB2GRAY)

        if len(predictor.shape) > 2:
            set_predictors[i, ..., 0] = np.copy(predictor[..., 0])
        else:
            set_predictors[i, ..., 0] = np.copy(predictor)

        set_predictors[i, ..., 0] /= 255
        sys.stdout.write("\rImage %i loaded" % i)

    return set_predictors
