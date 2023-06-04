import tensorflow as tf

from src.clearer.models.clearer_model_u_net import clearer_model_u_net
from src.clearer.trainers.trainer import train_clearer_model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if __name__ == "__main__":
    model_path = 'models/clearer/logs/clearer_weights.h5'
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.random.set_seed(2202)
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()
    model = clearer_model_u_net()
    train_clearer_model(model)
    model.save(model_path)
