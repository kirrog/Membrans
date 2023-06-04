from datetime import datetime

import tensorflow as tf

from src.utils.config_loader import train_data
from src.utils.config_loader import test_data
from src.clearer.dataset_generators.dataset_generator_providers import clearer_dataset_pair_creater

log_dir = 'models/clearer/logs/'
batch_size = 4
epochs = 50

optimizer = {
    "adam": tf.keras.optimizers.Adam(learning_rate=0.001),
    "sgd_nesterov": tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.01, nesterov=True, name="SGD"
    ),
    "sgd": tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD"
    )
}


def train_clearer_model(model):
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=1, monitor='loss', mode='min', min_delta=0.1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-accuracy{accuracy:.3f}_' + datetime.now().strftime(
                "%Y%m%d-%H%M%S") + '.h5',
            monitor='loss', mode='min')
    ]
    # -loss{loss:.3f}-val_loss{val_loss:.3f}_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer["adam"],
                  metrics=['accuracy'],
                  run_eagerly=True)

    train_dataset = clearer_dataset_pair_creater(train_data)
    test_dataset = clearer_dataset_pair_creater(test_data)
    print("Data created")
    model.fit(train_dataset, batch_size=batch_size,
              epochs=epochs,
              callbacks=my_callbacks,
              validation_data=test_dataset)
