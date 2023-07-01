import argparse
import time
from datetime import datetime
from pathlib import Path

from src.model.clearer import Clearer
from src.model.converter import Converter
from src.model.detector import Detector
from src.modules.collector import Collector
from src.modules.config import *
from src.modules.logger import CstmLogger
from src.modules.saver import Saver
from src.modules.stl_converter import StlConverter

img_x, img_y = 512, 512


def parse_args():
    parser = argparse.ArgumentParser("Аргументы программы")
    parser.add_argument("--input_directory", help="Путь до директории с файлом .txt и директорией dicom серии", type=str,
                        default="dataset/inference_data_dir/vnkv")
    parser.add_argument("--output_directory", help="Директория для сохранения результатов", type=str,
                        default="dataset/inference_results/vnkv_3")
    return parser.parse_args()


def main_pipeline(args, cstm_logger: CstmLogger):
    input_directory = args.input_directory
    output_directory = args.output_directory
    cstm_logger.log(f"Директория исходников: {input_directory}")
    cstm_logger.log(f"Директория результатов: {output_directory}")
    collector = Collector(cstm_logger)
    tomography_datacube, numbers_of_interest = collector.apply(input_directory)
    clearer = Clearer(clearer_model_path, cstm_logger)
    segmented_datacube = clearer.apply(tomography_datacube, batch_size)
    del clearer
    detector = Detector(detector_model_path, cstm_logger)
    segmented_datacube, defect_datacube = detector.apply(tomography_datacube, segmented_datacube, numbers_of_interest,
                                                         batch_size)
    del detector
    converter = Converter(converter_model_path, cstm_logger)
    membran_datacube = converter.apply(segmented_datacube, defect_datacube, batch_size)
    del converter
    stl_converter = StlConverter(cstm_logger)
    stl_surface = stl_converter.apply(membran_datacube)
    saver_obj = Saver(cstm_logger)
    saver_obj.save_all(segmented_datacube, defect_datacube, membran_datacube, stl_surface, Path(output_directory))


if __name__ == "__main__":
    start_time = time.time()
    start_date = datetime.now().strftime("%Y%m%d-%H%M%S")
    args = parse_args()
    with open(f"logs/{start_date}.logs", "w") as f:
        cstm_logger = CstmLogger(f)
        main_pipeline(args, cstm_logger)
    last_time = time.time()
    print(f"Время выполнения: {last_time - start_time:0.3f}")
