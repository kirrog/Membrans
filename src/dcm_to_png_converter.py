from pathlib import Path

import pydicom as dicom
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

nifti_dicom_path = "/media/kirrog/workdata/membransdata/newes"

root_path = Path(nifti_dicom_path)
subfilders = [x for x in root_path.glob("*") if x.is_dir()]
print(len(subfilders))

for subf in subfilders:
    dicom_p = subf / "dicom"
    csv = subf / "numbers.csv"
    dicom_orig = dicom_p / (subf.name + "O")
    dicom_membr = dicom_p / (subf.name + "M")
    dicom_bone = dicom_p / (subf.name + "B")
    d_l_o = []
    d_l_m = []
    d_l_b = []
    if not dicom_p.exists():
        print("dcm don't exists")
        exit(0)
    if not csv.exists():
        print("csv don't exists")
        exit(0)
    if not dicom_orig.exists():
        print("orig don't exists")
        exit(0)
    else:
        d_l_o = list(dicom_orig.glob("*.dcm"))
        d_l_o.sort()
        print(f"orig: len {len(d_l_o)}")
    if not dicom_membr.exists():
        print("membr don't exists")
        exit(0)
    else:
        d_l_m = list(dicom_membr.glob("*.dcm"))
        d_l_m.sort()
        print(f"orig: len {len(d_l_m)}")
    if not dicom_bone.exists():
        print("bone don't exists")
        exit(0)
    else:
        d_l_b = list(dicom_bone.glob("*.dcm"))
        d_l_b.sort()
        print(f"orig: len {len(d_l_b)}")
    assert len(d_l_o) == len(d_l_m) and len(d_l_b) == len(d_l_m)
    ng = subf / "NG"
    ng.mkdir(parents=True, exist_ok=True)
    r = subf / "R"
    r.mkdir(parents=True, exist_ok=True)
    g = subf / "G"
    g.mkdir(parents=True, exist_ok=True)
    rg = subf / "RG"
    rg.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(len(d_l_o))):
        ds = dicom.dcmread(str(d_l_o[i]))
        image_2d = ds.pixel_array.astype(float)
        image_2d[image_2d < 0.0] = 0.0
        image_2d_scaled_o = np.uint8((image_2d / image_2d.max()) * 255.0)
        ds = dicom.dcmread(str(d_l_m[i]))
        image_2d = ds.pixel_array.astype(float)
        image_2d[image_2d < 0.0] = 0.0
        image_2d_scaled_m = np.uint8((image_2d / image_2d.max()) * 255.0)
        ds = dicom.dcmread(str(d_l_b[i]))
        image_2d = ds.pixel_array.astype(float)
        image_2d[image_2d < 0.0] = 0.0
        image_2d_scaled_b = np.uint8((image_2d / image_2d.max()) * 255.0)

        ng_np = np.zeros((image_2d.shape[0], image_2d.shape[1], 4), dtype=np.uint8)
        ng_np[:, :, 1] = image_2d_scaled_o
        ng_np[:, :, 3] = image_2d_scaled_o
        plt.imsave((ng / f'{i:04d}.png'), ng_np)

        g_np = np.zeros((image_2d.shape[0], image_2d.shape[1], 4), dtype=np.uint8)
        g_np[:, :, 1] = image_2d_scaled_b
        g_np[:, :, 3] = image_2d_scaled_b
        plt.imsave((g / f'{i:04d}.png'), g_np)

        r_np = np.zeros((image_2d.shape[0], image_2d.shape[1], 4), dtype=np.uint8)
        r_np[:, :, 1] = image_2d_scaled_m
        r_np[:, :, 3] = image_2d_scaled_m
        plt.imsave((r / f'{i:04d}.png'), r_np)

        rg_np = np.zeros((image_2d.shape[0], image_2d.shape[1], 4), dtype=np.uint8)
        rg_np[:, :, 0] = image_2d_scaled_m
        rg_np[:, :, 1] = image_2d_scaled_b
        mask = image_2d_scaled_b + image_2d_scaled_m
        mask[mask != 0] = 255
        rg_np[:, :, 3] = mask
        plt.imsave((rg / f'{i:04d}.png'), rg_np)
