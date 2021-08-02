# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras

from src.clearer.clearer_model import clearer_model_new


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

if tf.test.gpu_device_name():
    print('GPU found: ' + tf.test.gpu_device_name())
else:
    print("No GPU found")

# def dice_metr(thresh):
#     smooth = 1e-3
#     def dice_coef(y_true, y_pred):
#         y_pred = (tf.sign(y_pred - thresh) + 1) / 2
#         y_true_f = keras.flatten(y_true)
#         y_pred_f = keras.flatten(y_pred)
#         intersection = keras.sum(y_true_f * y_pred_f)
#         return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

#     return dice_coef
