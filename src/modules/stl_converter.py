import voxelfuse as vf

from src.modules.logger import CstmLogger


class StlConverter:
    def __init__(self, cstm_logger: CstmLogger):
        self.cstm_logger = cstm_logger

    def apply(self, defect_datacube):
        self.cstm_logger.log("Создание воксельной модели")
        voxel_model = vf.VoxelModel(defect_datacube, 1)
        self.cstm_logger.log("Начало конвертации в полигоны")
        result = vf.Mesh.fromVoxelModel(voxel_model, (0.0, 1.0, 0.0, 1.0))
        self.cstm_logger.log("Конвертация выполнена")
        return result
