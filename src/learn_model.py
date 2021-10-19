import tensorflow as tf

from src.clearer.models.clearer_model_attention_u_net import clearer_model_attention_u_net
from src.clearer.trainers.trainer import train_clearer_model

model_path = '../models/clearer/logs/clearer_weights.h5'

tf.random.set_seed(2202)

model = clearer_model_attention_u_net()
train_clearer_model(model)
model.save(model_path)
