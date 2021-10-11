from keras.models import Model
from tensorflow import keras

from src.clearer.datasets_loader import load_clearer_dataset_tests
from src.clearer.models.clearer_model_attention_u_net import clearer_model_attention_u_net
from src.utils.matrix2png_saver import transform_results

model_path = '../models/clearer/clearer_weights.h5'
model_path_logs = '../models/clearer/logs/attention/ep002-loss0.136-val_loss0.278_20211011-233730.h5'
output_path = 'attention_2_l/'

logs = True
if logs:
    model = clearer_model_attention_u_net()
    model.load_weights(model_path_logs)
else:
    model = keras.models.load_model(model_path)

predicts = load_clearer_dataset_tests()
res = model.predict(x=predicts, batch_size=2)
transform_results(res, output_path)
