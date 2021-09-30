import tensorflow as tf

from src.clearer.clearer_model import clearer_model_new_result, clearer_model_test
from src.clearer.trainer import train_clearer_model


model_path = '../models/clearer/clearer_weights.h5'

tf.random.set_seed(2202)

model = clearer_model_test()
train_clearer_model(model)
model.save(model_path)
