from tensorflow import keras

from src.gen_3d.dataset_generator_provider.dataset_provider_test_detector import apply_model
from src.utils.config_loader import valid_data

model_path = "/home/kirrog/Documents/projects/Membrans/models/detector/logs/ep002-loss0.206-val_loss0.485-accuracy0.910_20220831-162646.h5"


model = keras.models.load_model(model_path)
model.summary()
batch_size = 10

apply_model(model, valid_data, "results")