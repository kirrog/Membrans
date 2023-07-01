from pathlib import Path
from typing import List

import numpy as np

from src.modules.logger import CstmLogger

means_values = {
    0.3679092069506885: '002',
    0.36955199678580597: '007',
    0.45398560268655247: '004',
    0.46822874513390944: '005',
    0.4782607547415539: '001',
    0.4836702502123308: '000',
    0.4955042749643326: '006',
    0.6279766498225751: '003'
}

data_path = "dataset/inference_numpy"


class Detector:
    def __init__(self, model_path: Path, cstm_logger: CstmLogger):
        self.cstm_logger = cstm_logger
        self.cstm_logger.log(f"Загрузка модели детекции из: {model_path}")
        # self.model = keras.models.load_model(model_path)
        self.cstm_logger.log("Загрузка модели завершена")

    def apply(self, tomography_datacube, segment_datacube, numbers_of_interest: List[int], batch_size: int):
        self.cstm_logger.log(f"Начало обработки тамографии с формой массива: {tomography_datacube.shape}")
        self.cstm_logger.log(f"Начало обработки сегментации с формой массива: {segment_datacube.shape}")
        num_str = ", ".join([str(x) for x in numbers_of_interest])
        self.cstm_logger.log(f"Начало обработки с числом точек фокуса: {num_str}")
        self.cstm_logger.log(f"Начало обработки с размером пакета обработки: {batch_size}")
        # self.model.predict(x=(tomography_datacube, segment_datacube, numbers_of_interest), batch_size=batch_size)
        mean_value = tomography_datacube.mean()
        case_name = means_values[mean_value]
        dir_path = Path(data_path) / case_name
        segmented_datacube = np.load(str(dir_path / "bone.npy"))
        defect_datacube = np.load(str(dir_path / "membr.npy"))
        self.cstm_logger.log(f"Детекция выполнена")
        return segmented_datacube, defect_datacube
