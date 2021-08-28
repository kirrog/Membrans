
# from clearer.datasets_loader import load_clearer_dataset_tests, load_clearer_dataset_predicts
# from utils.results_transformer import transform_results, check_results
# from clearer.clearer_model import clearer_model_new_deep

from src.clearer.datasets_loader import load_clearer_dataset_predicts
from src.utils.results_transformer import transform_results, check_results
from src.clearer.clearer_model import clearer_model_new_deep

output_path = 'test/'
model_path = '../models/clearer/logs/ep001.h5'


start = 100
step = 50
predicts = load_clearer_dataset_predicts()
model = clearer_model_new_deep()
model.load_weights(model_path)
table = predicts[start:(start+step)]
result = model.predict(table, batch_size=3)
check_results(result)
transform_results(result, output_path, start)