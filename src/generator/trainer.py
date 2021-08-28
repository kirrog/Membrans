import tensorflow as tf
from datetime import datetime

# from clearer.datasets_loader import load_clearer_dataset_predicts, load_clearer_dataset_answers
# from utils.augmentations import augment_dataset
# from utils.augment_dataset_generator import augment_size, get_augment_dataset

from src.generator.datasets_loader import load_generator_dataset_predicts, load_generator_dataset_answers
from src.utils.augmentations import augment_dataset
from src.utils.augment_dataset_generator import augment_size, get_augment_dataset

# log_dir = '/content/drive/MyDrive/Membrans/models/generator/logs/'

log_dir = '../models/generator/logs/'

def load_orig_dataset():
  answers_orig = load_generator_dataset_answers()
  predictors_orig = load_generator_dataset_predicts()
  return predictors_orig, answers_orig

def load_new_augment_dataset():
  answers_orig = load_generator_dataset_answers()
  predictors_orig = load_generator_dataset_predicts()
  return augment_dataset(predictors_orig, answers_orig)

def get_augment_dataset_size():
  return augment_size


def train_generator_model(model):
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=1, monitor='loss', mode='min', min_delta=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=log_dir + 'ep{epoch:03d}.h5',
                        monitor='loss', mode='min')
    ]
    #-loss{loss:.3f}-val_loss{val_loss:.3f}_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '

    batch_size = 3
    epochs = 50

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    answers = load_generator_dataset_answers()
    predictors = load_generator_dataset_predicts()

    model.fit(predictors, answers, batch_size=batch_size,
                        epochs=epochs, verbose=1,
                        validation_split=0.1,
                        callbacks=my_callbacks)
