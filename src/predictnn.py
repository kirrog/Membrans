import glob
import sys
import cv2
import pydicom as dicom
from natsort import natsorted
from tensorflow import keras
import numpy as np

from src.utils.config_loader import threads_loading
from src.utils.config_loader import batch_size
from src.utils.config_loader import model_path
from src.utils.config_loader import model_path_logs
from src.utils.config_loader import one_test_data_dir
from src.utils.config_loader import output_data_dir

img_x, img_y = 512, 512
model = keras.models.load_model(model_path)
model.summary()
exit(0)


def from_dcm_to_png(directory):
    paths_predicts = natsorted(glob.glob(directory + '*.dcm'))
    print(len(paths_predicts))
    pred = np.zeros([len(paths_predicts), 512, 512])
    dcm = 0
    for file, iter in zip(paths_predicts, range(len(paths_predicts))):
        dcm = dicom.dcmread(file)
        image_2d = dcm.pixel_array.astype(float)
        image_2d = (np.maximum(image_2d, 0) / image_2d.max())
        image_2d = cv2.resize(image_2d, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
        pred[iter, :, :] = image_2d
        sys.stdout.write("\rImage %i loaded" % iter)
    print(pred.shape[0])
    return pred, dcm


data, dcm = from_dcm_to_png(one_test_data_dir)
img_x_orig = dcm.Columns
img_y_orig = dcm.Rows
results = model.predict(x=data, batch_size=batch_size)


def from_dcm_to_png(directory, dcm, data, img_x_orig, img_y_orig):
    for iter in range(data.shape[0]):
        # '{:03}'.format(iter)
        dcm.PixelData = cv2.resize(data[iter, :, :], (img_x_orig, img_y_orig), interpolation=cv2.INTER_CUBIC)
        dicom.dcmwrite((directory + '{:04}'.format(iter) + '.dcm'), dcm, True)
        sys.stdout.write("\rImage %i written" % iter)

from_dcm_to_png(output_data_dir, dcm, results, img_x_orig, img_y_orig)