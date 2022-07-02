import tensorflow as tf
from datetime import datetime

from src.generator.dataset_generator_providers.dataset_generator_providers import generator_dataset_pair_creater
from src.utils.config_loader import train_data
from src.utils.config_loader import test_data
from src.clearer.dataset_generators.dataset_generator_providers import clearer_dataset_pair_creater

log_dir = '../models/generator/logs/'


def train_generator_model(model):
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=1, monitor='loss', mode='min', min_delta=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}_' + datetime.now().strftime(
                "%Y%m%d-%H%M%S") + '.h5',
            monitor='loss', mode='min')
    ]
    # -loss{loss:.3f}-val_loss{val_loss:.3f}_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '

    batch_size = 3
    epochs = 50

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    train_dataset = generator_dataset_pair_creater(train_data)
    test_dataset = generator_dataset_pair_creater(test_data)

    model.fit(train_dataset, batch_size=batch_size,
              epochs=epochs,
              callbacks=my_callbacks,
              validation_data=test_dataset)
