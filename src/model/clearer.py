from pathlib import Path

from tensorflow import keras

from src.modules.logger import CstmLogger


class Clearer:
    def __init__(self, model_path: Path, cstm_logger: CstmLogger):
        self.cstm_logger = cstm_logger
        self.cstm_logger.log(f"Загрузка модели очистки из: {model_path}")
        self.model = keras.models.load_model(model_path)
        self.cstm_logger.log(f"Загрузка модели завершена")

    def apply(self, tomography_datacube, batch_size):
        self.cstm_logger.log(f"Начало обработки тамографии формы: {tomography_datacube.shape} размер пакета обработки: {batch_size}")
        result = self.model.predict(x=tomography_datacube, batch_size=batch_size)
        self.cstm_logger.log(f"Очистка завершена")
        return result
