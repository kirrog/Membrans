import voxelfuse as vf

from src.modules.logger import CstmLogger


class StlConverter:
    def __init__(self, cstm_logger: CstmLogger):
        self.cstm_logger = cstm_logger

    def apply(self, defect_datacube):
        self.cstm_logger.log("Create voxel model")
        voxel_model = vf.VoxelModel(defect_datacube, 1)
        self.cstm_logger.log("Start conversion to stl")
        result = vf.Mesh.fromVoxelModel(voxel_model, (0.0, 1.0, 0.0, 1.0))
        self.cstm_logger.log("Conversion complete")
        return result
