import glob
import time

import cv2
import pydicom
import numpy as np
from natsort import natsorted

from pydicom.misc import is_dicom


def find_dicom_files(directory):
    dcm_files = []
    files = natsorted(glob.glob(directory + '/*.dcm'))
    for file in files:
        if is_dicom(file):
            dcm_files.append(file)
    return dcm_files


def convert_dicom(directory):
    dicom_files = find_dicom_files(directory)

    ref_data = pydicom.dcmread(dicom_files[0])
    out_shape = (len(dicom_files), int(ref_data.Rows), int(ref_data.Columns))
    print('Pixels data type: {}'.format(ref_data.pixel_array.dtype))  # probably always uint16
    print('Slice Thickness: {}'.format(ref_data.SliceThickness))  # on ref image
    print('Pixel Spacing: {}'.format(ref_data.PixelSpacing))  # on ref image
    # print(ref_data.suid)
    # print(ref_data.SOPClassUID) # for ct 1.2.840.10008.5.1.4.1.1.2
    # print(ref_data.SeriesInstanceUID)

    dicom_np_array = np.zeros(out_shape, dtype=ref_data.pixel_array.dtype)

    for dcm_file in dicom_files:
        data = pydicom.dcmread(dcm_file)
        # data.sort(key=lambda x: float(x.ImagePositionPatient[2]))

        if data.pixel_array.shape[0] < dicom_np_array.shape[1] or data.pixel_array.shape[1] < dicom_np_array.shape[2]:
            data.pixel_array = cv2.resize(data.pixel_array, (dicom_np_array.shape[2], dicom_np_array.shape[1]),
                                          interpolation=cv2.INTER_CUBIC)
        elif data.pixel_array.shape[0] > dicom_np_array.shape[1] or data.pixel_array.shape[1] > dicom_np_array.shape[2]:
            data.pixel_array = cv2.resize(data.pixel_array, (dicom_np_array.shape[2], dicom_np_array.shape[1]),
                                          interpolation=cv2.INTER_AREA)

        dicom_np_array[dicom_files.index(dcm_file), :, :] = data.pixel_array

    return dicom_np_array


if __name__ == '__main__':
    test_dir = ''
    start = time.time()
    arr = convert_dicom(test_dir)
    print(arr.shape)
    end = time.time()
    print('Execution time: {}'.format(end-start))  # 7.8 sec for 361 dcm file
