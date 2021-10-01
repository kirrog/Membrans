import tensorflow as tf
from datetime import datetime
import numpy as np

from src.clearer.dataset_generator_providers import clearer_dataset_answers_generator, \
    clearer_dataset_predicts_generator, clearer_dataset_pair_generator, \
    clearer_dataset_pair_creater
from src.clearer.datasets_loader import load_clearer_dataset_predicts, load_clearer_dataset_answers
from src.utils.augmentations import augment_dataset
from src.utils.augment_dataset_generator import augment_size, get_augment_dataset

log_dir = '../models/clearer/logs/'
batch_size = 3
epochs = 50


def load_orig_dataset():
    answers_orig = load_clearer_dataset_answers()
    predictors_orig = load_clearer_dataset_predicts()
    return predictors_orig, answers_orig


def load_new_augment_dataset():
    answers_orig = load_clearer_dataset_answers()
    predictors_orig = load_clearer_dataset_predicts()
    return augment_dataset(predictors_orig, answers_orig)


def get_augment_dataset_size():
    return augment_size


def train_clearer_model(model):
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='loss', mode='min', min_delta=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=log_dir + 'ep{epoch:03d}.h5',
                                           monitor='loss', mode='min')
    ]
    # -loss{loss:.3f}-val_loss{val_loss:.3f}_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    train_dataset, test_dataset = clearer_dataset_pair_creater()

    model.fit(train_dataset, batch_size=batch_size,
              epochs=epochs,
              callbacks=my_callbacks,
              validation_data=test_dataset)
