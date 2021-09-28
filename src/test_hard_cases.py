from tensorflow import keras

# from clearer.datasets_loader import load_clearer_dataset_tests
# from clearer.results_transformer import transform_results, check_results

from src.clearer.datasets_loader import load_clearer_dataset_tests
from src.utils.matrix2png_saver import transform_results, check_results

model_path = '../models/clearer/clearer_weights.h5'
output_path = 'test/'


model = keras.models.load_model(model_path)
predicts = load_clearer_dataset_tests()
res = model.predict(predicts[50:60])
check_results(res)
transform_results(res, output_path)
