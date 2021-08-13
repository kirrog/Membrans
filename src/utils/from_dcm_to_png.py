import sys

import numpy as np
import pydicom as dicom
import cv2
import os
from pathlib import Path


# result_path = '/content/drive/MyDrive/Membrans/dataset/new/'

result_path = os.getcwd() + '\\..\\dataset\\new\\'

def from_dcm_to_png(directory):
    p = Path(directory)
    files = [x for x in p.iterdir() if x.is_file()]
    for file,iter in zip(files, range(1, len(files) + 1)):
        read_write_file(file, '{:03}'.format(iter))


def read_write_file(file_path, iter):
    ds = dicom.dcmread(file_path)

    image_2d = ds.pixel_array.astype(float)
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    image_2d_scaled = np.uint8(image_2d_scaled)

    image_format = '.png'
    image_path = result_path + (iter) + image_format

    cv2.imwrite(image_path, image_2d_scaled)
    sys.stdout.write("\rImage %i transformed to png" % iter)