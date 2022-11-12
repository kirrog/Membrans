import tensorflow as tf

from src.gan_model.model.generator_model_u_net import *
from src.gan_model.trainer.trainer import train_generator_model, test_generator_model

tf.random.set_seed(2202)

model_co = generator_model_coder()
model_de = generator_model_decoder()
model_di = generator_model_discriminator()

train_generator_model(model_co, model_de, model_di)
test_generator_model(model_co, model_de, model_di)

