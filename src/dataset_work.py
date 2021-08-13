from tensorflow import keras

import numpy as np

from src.clearer.datasets_loader import load_clearer_dataset_tests
from src.clearer.results_transformer import transform_results


def iter(filt, predicts):
    size = len(predicts)
    model_path = '..\\models\\clearer\\clearer_weights_' + str(filt) + '.h5'
    model = keras.models.load_model(model_path)
    results_list = []
    list = np.array_split(predicts, int(size/10))
    for table in list:
        res = model.predict(table)
        results_list.append(res)
    result = np.concatenate(results_list)
    transform_results(result, str(filt) + '\\')


predicts = load_clearer_dataset_tests()
iter(8, predicts)
iter(16, predicts)
iter(32, predicts)
iter(64, predicts)
