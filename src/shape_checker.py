import argparse
from pathlib import Path

import pydicom as dicom
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Аргументы программы")
    parser.add_argument("--input_directory", help="Путь до директории с файлом .txt и директорией dicom серии",
                        type=str,
                        default="dataset/inference_data_dir/vnkv")
    return parser.parse_args()


class Cheker:
    img_x_size: int = 512
    img_y_size: int = 512

    def check_shapes(self, directory: Path):
        paths_predicts = list(sorted(directory.glob("*")))
        print(f"Number of layers: {len(paths_predicts)}")
        sizes_x = set()
        sizes_y = set()
        for filepath2dicom in paths_predicts:
            dcm = dicom.dcmread(filepath2dicom)
            image_2d_numpy = dcm.pixel_array.astype(float)
            size_x, size_y = image_2d_numpy.shape
            sizes_x.add(size_x)
            sizes_y.add(size_y)
        if len(sizes_x) > 1 or len(sizes_y) > 1:
            print(f"[ошибка] dicom серия обладает изображениями разных размеров:")
            print(f"[ошибка] ширина: {sizes_x}")
            print(f"[ошибка] высота: {sizes_y}")
            return False
        s_x = sizes_x.pop()
        s_y = sizes_y.pop()
        if s_x != self.img_x_size or s_y != self.img_y_size:
            print(f"[ошибка] dicom серия обладает размерами отличными от приемлемых: примемлимые: "
                  f"{self.img_x_size}х{self.img_y_size} имеющиеся: {s_x}х{s_y}")

    def apply(self, input_directory: str):
        dir_path = Path(input_directory)
        if not dir_path.exists():
            print(f"[ошибка] Директория не существует: {input_directory}")
            exit(1)
        dicom_path = dir_path / "DICOM"
        if not dicom_path.exists():
            print(f"[ошибка] Директория dicom не существует: {str(dicom_path)}")
            exit(1)
        return self.check_shapes(dicom_path)


if __name__ == "__main__":
    args = parse_args()
    check = Cheker()
    check.apply(args.input_directory)
