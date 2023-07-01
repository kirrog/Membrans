from pathlib import Path
from pprint import pprint

import cv2
import numpy as np
import pydicom as dicom
from tqdm import tqdm

from src.modules.logger import CstmLogger


class Collector:
    img_x_size: int = 512
    img_y_size: int = 512

    def __init__(self, cstm_logger: CstmLogger):
        self.cstm_logger = cstm_logger

    def _from_dcm_to_png(self, directory: Path):
        paths_predicts = list(sorted(directory.glob("*")))
        self.cstm_logger.log(f"Number of layers: {len(paths_predicts)}")
        loaded_data = np.zeros([len(paths_predicts), 512, 512])
        for i, filepath2dicom in tqdm(enumerate(paths_predicts)):
            dcm = dicom.dcmread(filepath2dicom)
            image_2d_numpy = dcm.pixel_array.astype(float)
            image_2d_normalised = np.maximum(image_2d_numpy, 0)
            image_2d_resized = cv2.resize(image_2d_normalised, (self.img_x_size, self.img_y_size),
                                          interpolation=cv2.INTER_CUBIC)
            loaded_data[i, :, :] = image_2d_resized
        loaded_data_normalised = loaded_data / min(loaded_data.max(), loaded_data.mean() * 2)
        loaded_data_normalised[loaded_data_normalised > 0.0] = 1.0
        loaded_data_normalised[loaded_data_normalised < 0.0] = 0.0
        self.cstm_logger.log(f"Форма датакуба: {loaded_data_normalised.shape}")
        return loaded_data_normalised

    def apply(self, input_directory: str):
        dir_path = Path(input_directory)
        if not dir_path.exists():
            self.cstm_logger.log(f"[ошибка] Директория не существует: {input_directory}")
            exit(1)
        file_path = dir_path / "numbers.csv"
        if not file_path.exists():
            self.cstm_logger.log(f"[ошибка] Файл точек интереса не существует: {str(file_path)}")
            exit(1)
        try:
            with open(str(file_path), "r") as f:
                numbers = [int(x) for x in f.read().split(",")]
        except Exception as e:
            self.cstm_logger.log("[ошибка] Невозможно загрузить точки интереса")
            pprint(e)
            exit(1)
        dicom_path = dir_path / "DICOM"
        if not dicom_path.exists():
            self.cstm_logger.log(f"[ошибка] Директория dicom не существует: {str(dicom_path)}")
            exit(1)
        dicom_image = self._from_dcm_to_png(dicom_path)
        return dicom_image, numbers
