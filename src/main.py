import argparse
import json
from multiprocessing.pool import Pool
from pathlib import Path
from pprint import pprint

import cv2
import nibabel as nib
import numpy as np
import pydicom as dicom
from tqdm import tqdm

from src.clearer.dataset_generators.dataset_generator_providers import clearer_dataset_pair_creater
from src.utils.config_loader import train_data

patient_path = '../patient_data'
clearer_model_path = '../result_models/clearer_weights.h5'
generator_model_path = '../result_models/generator_weights.h5'
teeth_finder_model_path = '../result_models/teeth_finder_weights.h5'
numpys_path = '../patient_data/numpys/'
paths_pred_masks = '/*_image/*.png'

batch_size = 3
img_x, img_y = 512, 512


def parse_args():
    parser = argparse.ArgumentParser("Program arguments")
    parser.add_argument("--input_directory", help="Directory of dicom dir and txt", type=str)
    parser.add_argument("--output_directory", help="Directory of stl", type=str)
    parser.add_argument("--config", help="Config file", type=str, default="../inference_config.json")
    args = parser.parse_args()
    return args


def parse_config_file(config_file: str):
    path = Path(config_file)
    if not path.exists():
        print(f"Config file dn exists {config_file}")
        exit(1)
    with open(config_file, "r") as f:
        data = json.load(f)
        pprint(data)
    return data


def load_dicom_image(filepath2dicom: Path):
    dcm = dicom.dcmread(filepath2dicom)
    instance_number = int(dcm.InstanceNumber)
    image_2d_numpy = dcm.pixel_array.astype(float)
    image_2d_normalised = (np.maximum(image_2d_numpy, 0) / image_2d_numpy.max())
    image_2d_resized = cv2.resize(image_2d_normalised, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
    return image_2d_resized, instance_number


def from_dcm_to_png(directory: Path):
    paths_predicts = sorted(directory.glob("*"))
    print(len(paths_predicts))
    loaded_data = np.zeros([len(paths_predicts), 512, 512])
    for i, filepath2dicom in tqdm(enumerate(paths_predicts)):
        dcm = dicom.dcmread(filepath2dicom)
        image_2d_numpy = dcm.pixel_array.astype(float)
        image_2d_normalised = (np.maximum(image_2d_numpy, 0) / image_2d_numpy.max())
        image_2d_resized = cv2.resize(image_2d_normalised, (img_x, img_y), interpolation=cv2.INTER_CUBIC)
        loaded_data[i, :, :] = image_2d_resized
    # with Pool() as p:
    #     for image_2d, i in tqdm(p.imap(load_dicom_image, paths_predicts)):
    #         loaded_data[i, :, :] = image_2d
    print(loaded_data.shape[0])
    return loaded_data


def load_patient(dir_path_str: str):
    dir_path = Path(dir_path_str)
    if not dir_path.exists():
        print(f"Input dir dn exists {dir_path_str}")
        exit(1)
    file_path = dir_path / "numbers.csv"
    if not file_path.exists():
        print(f"Input features file dn exists {str(file_path)}")
        exit(1)
    try:
        with open(str(file_path), "r") as f:
            numbers = [int(x) for x in f.read().split(",")]
    except Exception as e:
        print("Can't load info from features file")
        pprint(e)
        exit(1)
    dicom_path = dir_path / "DICOM"
    if not dicom_path.exists():
        print(f"Input dicom dir file dn exists {str(file_path)}")
        exit(1)
    dicom_image = from_dcm_to_png(dicom_path)
    return dicom_image, numbers


if __name__ == "__main__":
    # args = parse_args()
    # input_directory = args.input_directory if args.input_directory is not None else "../dataset/inference_data_dir/001"
    # output_directory = args.output_directory if args.output_directory is not None else "../dataset/inference_results/001"
    # config = args.config
    # config_data = parse_config_file(config)
    # numpy_image, nums = load_patient(input_directory)
    exit(0)
