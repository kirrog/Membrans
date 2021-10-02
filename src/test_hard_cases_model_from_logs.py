
from src.clearer.datasets_loader import load_clearer_dataset_predicts
from src.utils.matrix2png_saver import transform_results, check_results
from src.clearer.models.clearer_model_u_net import clearer_model_new

output_path = 'test/'
model_path = '../models/clearer/logs/adam/ep001.h5'


start = 100
step = 50
predicts = load_clearer_dataset_predicts()
model = clearer_model_new()
model.load_weights(model_path)
table = predicts[start:(start+step)]
result = model.predict(table, batch_size=3)
check_results(result)
transform_results(result, output_path, start)