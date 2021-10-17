from tensorflow import keras

from src.clearer.datasets_loader import load_clearer_dataset_tests, load_test_dataset_from_images_from
from src.clearer.models.clearer_model_attention_u_net import clearer_model_attention_u_net
from src.clearer.models.clearer_model_u_net import clearer_model_u_net
from src.utils.matrix2png_saver import transform_results

model_path = '../models/clearer/clearer_weights.h5'
model_path_logs = '../models/clearer/logs/u_a/ep002-loss0.087-val_loss0.156_20211015-032207.h5'
output_path = 'case/ktv/'

logs = True
if logs:
    model = clearer_model_u_net()
    model.load_weights(model_path_logs)
else:
    model = keras.models.load_model(model_path)

predicts = load_test_dataset_from_images_from("../dataset/test_cases/ktv/")
res = model.predict(x=predicts, batch_size=2)
transform_results(res, output_path)
