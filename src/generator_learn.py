import tensorflow as tf

from src.generator.model.generator_model_u_net import generator_model_u_net
from src.generator.trainer.trainer import train_generator_model
from src.utils.config_loader import model_path

tf.random.set_seed(2202)

model = generator_model_u_net()
train_generator_model(model)
model.save(model_path)
