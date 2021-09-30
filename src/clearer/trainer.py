import tensorflow as tf
from datetime import datetime
import numpy as np

from src.clearer.dataset_generator_providers import clearer_dataset_answers_generator, \
    clearer_dataset_predicts_generator, clearer_dataset_pair
from src.clearer.datasets_loader import load_clearer_dataset_predicts, load_clearer_dataset_answers
from src.utils.augmentations import augment_dataset
from src.utils.augment_dataset_generator import augment_size, get_augment_dataset

log_dir = '../models/clearer/logs/'


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


def train_clearer_model_augment(model):
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=1, monitor='loss', mode='min', min_delta=1000),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir + 'ep{epoch:03d}.h5',
            monitor='loss', save_weights_only=True, save_best_only=True, mode='min')
    ]

    batch_size = 3
    epochs = 50

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    answers = load_clearer_dataset_answers()

    iters = get_augment_dataset_size()

    for i in range(iters):
        predictors = get_augment_dataset(i)
        print('\nUse ' + str(i) + ' dataset to train')
        model.fit(predictors, answers, batch_size=batch_size,
                  epochs=epochs, verbose=1,
                  validation_split=0.1,
                  callbacks=my_callbacks)


def train_clearer_model(model):
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='loss', mode='min', min_delta=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=log_dir + 'ep{epoch:03d}.h5',
                                           monitor='loss', mode='min')
    ]
    # -loss{loss:.3f}-val_loss{val_loss:.3f}_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '

    batch_size = 3
    epochs = 50

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    dataset = tf.data.Dataset.from_generator(
        lambda: next(clearer_dataset_pair()),
        output_types=np.float32, output_shapes=(batch_size, 512, 512))

    # Splitting the dataset for training and testing.
    def is_test(x, _):
        return x % 4 == 0

    def is_train(x, y):
        return not is_test(x, y)

    recover = lambda x, y: y

    # Split the dataset for training.
    test_dataset = dataset.enumerate() \
        .filter(is_test) \
        .map(recover)

    # Split the dataset for testing/validation.
    train_dataset = dataset.enumerate() \
        .filter(is_train) \
        .map(recover)

    model.fit(train_dataset, batch_size=batch_size,
              epochs=epochs, verbose=1,
              callbacks=my_callbacks,
              validation_data=test_dataset)
