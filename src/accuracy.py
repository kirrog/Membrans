import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("Аргументы программы")
    parser.add_argument("--prediction_file", help="Путь до файла массива предсказанной мембраны", type=str,
                        default="dataset/inference_results/vnkv/membran.npy")
    parser.add_argument("--original_file", help="Путь до файла массива оригинальной мембраны", type=str,
                        default="dataset/inference_results/vnkv_3/membran.npy")
    return parser.parse_args()


def shape_size(shape):
    result = 1
    for num in shape:
        result *= num
    return result


def compare_shape(shape_0, shape_1):
    result = True
    result &= len(shape_0) == len(shape_1)
    if not result:
        return result
    for i, j in zip(shape_0, shape_1):
        result &= i == j
    return result


if __name__ == "__main__":
    args = parse_args()
    predicted = np.load(args.prediction_file)
    original = np.load(args.original_file)
    if not compare_shape(original.shape, predicted.shape):
        print(f"[Ошибка] Неверная форма массива в файлах: предсказанный массив: {predicted.shape} "
              f"оригинальный массив: {original.shape}")
        exit(1)
    result = 1 - (np.sum(np.abs(original - predicted)) / shape_size(original.shape))
    print(f"Точность: {result:0.9f}")
