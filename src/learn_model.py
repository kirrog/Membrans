import tensorflow as tf

from src.clearer.clearer_model import clearer_model_new_result, clearer_model_test, clearer_model_new
from src.clearer.trainer import train_clearer_model
from src.generator.generator_model import generator_model_new
from src.generator.trainer import train_generator_model

model_path = '../models/generator/clearer_weights.h5'

tf.random.set_seed(2202)

model = generator_model_new()
train_generator_model(model)
model.save(model_path)
