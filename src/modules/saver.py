from pathlib import Path

import numpy as np

from src.modules.logger import CstmLogger


class Saver:
    segmentation_template = "segmentation.npy"
    defect_template = "defect.npy"
    membran_template = "membran.npy"
    stl_template = "membran.stl"

    def __init__(self, cstm_logger: CstmLogger):
        self.cstm_logger = cstm_logger

    def save_all(self, segmented_datacube, defect_datacube, membran_datacube, stl_surface, output_directory: Path):
        output_directory.mkdir(parents=True, exist_ok=True)
        segmentation_res_path = output_directory / self.segmentation_template
        defect_res_path = output_directory / self.defect_template
        membran_res_path = output_directory / self.membran_template
        stl_res_path = output_directory / self.stl_template
        np.save(str(segmentation_res_path), segmented_datacube)
        self.cstm_logger.log(f"Segmentation results saved at: {segmentation_res_path}")
        np.save(str(defect_res_path), defect_datacube)
        self.cstm_logger.log(f"Defect results saved at: {defect_res_path}")
        np.save(str(membran_res_path), membran_datacube)
        self.cstm_logger.log(f"Membran results saved at: {membran_res_path}")
        stl_surface.export(str(stl_res_path))
        self.cstm_logger.log(f"Stl results saved at: {stl_res_path}")
