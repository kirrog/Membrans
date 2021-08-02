import tensorflow as tf
from src.clearer.clearer_model import clearer_model_new, save_model
from src.clearer.trainer import train_clearer_model


tf.random.set_seed(2020)

model = clearer_model_new()

train_clearer_model(model)

save_model(model)