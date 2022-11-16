from tensorflow import keras

import numpy as np

from src.gen_3d.dataset_generator_provider.dataset_provider_test_detector import apply_model
from src.gen_3d.model.generator_model_detector import detector_model_getter
from src.gen_3d.trainer.detector_trainer import train_detector_model
from src.utils.config_loader import test_data, valid_data, train_data, model_path

model = keras.models.load_model(model_path)
#model = detector_model_getter()
#train_detector_model(model)

# train_dataset = generator_dataset_pair_creater(train_data)
# test_dataset = generator_dataset_pair_creater(test_data)
out_path = "/home/kirrog/projects/Membrans/dataset/3d_results"
apply_model(model, test_data, out_path)