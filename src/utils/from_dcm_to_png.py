import glob
import sys

import numpy as np
import pydicom as dicom
import cv2

result_path = '../dataset/new/'


def from_dcm_to_png(directory):
    paths_predicts = sorted(glob.glob(directory + '*.dcm'))
    for file, iter in zip(paths_predicts, range(1, len(paths_predicts) + 1)):
        read_write_file(file, '{:03}'.format(iter))


def read_write_file(file_path, iter):
    ds = dicom.dcmread(file_path)

    image_2d = ds.pixel_array.astype(float)
    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
    image_2d_scaled = np.uint8(image_2d_scaled)

    image_format = '.png'
    image_path = result_path + (iter) + image_format

    cv2.imwrite(image_path, image_2d_scaled)
    sys.stdout.write("\rImage %s transformed to png" % iter)
