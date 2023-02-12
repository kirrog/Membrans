import json
from pathlib import Path

import numpy as np
import argparse
import nibabel as nib
import dicom2nifti

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
    parser.add_argument("--input_directory", help="Directory of dicom", type=str)
    parser.add_argument("--input_file", help="File with places if interest", type=str)
    parser.add_argument("--output_directory", help="Directory of stl", type=str)
    parser.add_argument("--cache_directory", help="Directory of cache", type=str)
    parser.add_argument("--config", help="Config file", type=str, default="./config.json")
    args = parser.parse_args()
    return args


def parse_config_file(config_file: str):
    path = Path(config_file)
    if not path.exists():
        print(f"Config file dn exists {config_file}")
        exit(1)
    with open(config_file, "r") as f:
        data = json.load(f)
        print(data)
    return data


def convert_dcm2nifti(input_dir: Path, output_nifti: Path):
    dicom2nifti.dicom_series_to_nifti(str(input_dir), str(output_nifti))


def load_nifti(nifti_file_path: Path):
    return nib.load(str(nifti_file_path))


def load_patient(dir_path_str: str, file_path_str: str, cache_dir_path_str: str):
    dir_path = Path(dir_path_str)
    if not dir_path.exists():
        print(f"Input directory dn exists {dir_path_str}")
        exit(1)
    file_path = Path(file_path_str)
    if not file_path.exists():
        print(f"Input file dn exists {file_path_str}")
        print("Can't get features")
    cache_dir_path = Path(cache_dir_path_str)
    cache_dir_path.mkdir(exist_ok=True, parents=True)
    nifti_output_file = cache_dir_path / "in_data.nii.gz"
    convert_dcm2nifti(dir_path, nifti_output_file)
    nifti_image = load_nifti(nifti_output_file)
    with open(file_path_str, "r") as f:
        nums = f.read().split(",")
    print(nums)
    return nifti_image, nums


def clearer_predict(config: dict, input_data: np.array) -> np.array:
    return None


def generator_predict(config: dict, input_data: np.array) -> np.array:
    return None


def get_membran_surface_from_volume(config: dict, input_data: np.array) -> np.array:
    return None


def convert_3d_array2stl(config: dict, input_data: np.array):
    return 1


def save2nifti(path_to_save: Path, data: nib.nifti2.Nifti1Image):
    nib.save(data, str(path_to_save))


def save_stl(path_to_save: Path, stl_data):
    pass


args = parse_args()
input_directory = args.input_directory
input_file = args.input_file
output_directory = args.output_directory
cache_directory = args.cache_directory
config = args.config
config_data = parse_config_file(config)
nifti_image, nums = load_patient(input_directory, input_file, cache_directory)
nifti_data = nifti_image.get_data()
print(f"Patient data loaded: shape {nifti_data.shape} "
      f"min {nifti_data.min()} max {nifti_data.max()} mean {nifti_data.mean()}")
cleared_data = clearer_predict(config_data, nifti_data)
save2nifti(cache_directory / "cleared.nii.gz", cleared_data)
generated_data = generator_predict(config_data, cleared_data)
save2nifti(cache_directory / "generated.nii.gz", generated_data)
surface_data = get_membran_surface_from_volume(config_data, generated_data)
save2nifti(cache_directory / "surface.nii.gz", surface_data)
stl_data = convert_3d_array2stl(config_data, surface_data)
save_stl(cache_directory / "membran_surface.stl", stl_data)
