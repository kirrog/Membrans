import tensorflow as tf

# from clearer.clearer_model import clearer_model_new_resul
# from clearer.trainer import train_clearer_model

from src.clearer.clearer_model import clearer_model_new_resul
from src.clearer.trainer import train_clearer_model

# model_path = '/content/drive/MyDrive/Membrans/models/clearer/clearer_weights.h5'

model_path = '\\..\\models\\clearer\\clearer_weights.h5'

tf.random.set_seed(2202)

model = clearer_model_new_resul()
train_clearer_model(model)

model.save(model_path)
