
import numpy as np

predict_path = '../dataset/numpys/generator/predictors.npy'
answers_path = '../dataset/numpys/generator/answers.npy'
tests_path = '../dataset/numpys/generator/test.npy'


def save_generator_dataset_tests(numpy_table):
    save_generator_dataset_to_numpy_table(numpy_table, tests_path)


def save_generator_dataset_answers(numpy_table):
    save_generator_dataset_to_numpy_table(numpy_table, answers_path)


def save_generator_dataset_predicts(numpy_table):
    save_generator_dataset_to_numpy_table(numpy_table, predict_path)


def save_generator_dataset_to_numpy_table(numpy_table, file_path):
    np.save(file_path, numpy_table)


def load_generator_dataset_tests():
    return load_generator_dataset_to_numpy_table(tests_path)


def load_generator_dataset_answers():
    return load_generator_dataset_to_numpy_table(answers_path)


def load_generator_dataset_predicts():
    return load_generator_dataset_to_numpy_table(predict_path)


def load_generator_dataset_to_numpy_table(file_path):
    return np.load(file_path)
