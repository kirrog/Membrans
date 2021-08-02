import numpy as np
import pydicom as dicom
import cv2
import os
from pathlib import Path

def from_dcm_to_png(direcotory):
    p = Path(os.getcwd() + '\\..\\' + direcotory)
    files = [x for x in p.iterdir() if x.is_file()]
    for file,iter in zip(files, range(1, len(files) + 1)):
        read_write_file(file, '{:03}'.format(iter))


def read_write_file(file_path, iter):
    ds = dicom.dcmread(file_path)

    image_2d = ds.pixel_array.astype(float)
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    image_2d_scaled = np.uint8(image_2d_scaled)

    image_format = '.png'
    cwd = os.getcwd()
    image_path = cwd + '\\..\\dataset\\new\\' + (iter) + image_format

    cv2.imwrite(image_path, image_2d_scaled)
    print("Written " + iter)