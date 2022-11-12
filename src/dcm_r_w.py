import pydicom as dicom
import numpy as np
from matplotlib import pyplot as plt
from pydicom.uid import RLELossless

dcm = dicom.dcmread('../dataset/test/001.dcm')

# dcm.PixelData = np.zeros([600, 600]).tobytes()
data = np.zeros([600, 600], dtype=np.uint16)
for i in range(600):
    data[i, 100] = 255.0

dcm.compress(RLELossless, data)
dicom.dcmwrite('../dataset/test/002.dcm', dcm, True)

dcm = dicom.dcmread('../dataset/test/002.dcm')
png = dcm.pixel_array / 255
res = np.zeros([600, 600, 3])
res[:, :, 1] = png  # [:, :, 0]
plt.imsave('../dataset/test/002.png', res)
