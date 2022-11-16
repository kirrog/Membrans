import logging
from datetime import datetime

import tensorflow as tf

from src.gen_3d.dataset_generator_provider.dataset_provider_detector import generator_dataset_pair_creater
from src.utils.config_loader import test_data, valid_data, train_data

log_dir = 'models/detector/logs/'
batch_size = 6
epochs = 1

optimizer = {
    "adam": 'adam',
    "sgd_nesterov": tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.01, nesterov=True, name="SGD"
    ),
    "sgd": tf.keras.optimizers.SGD(
        learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD"
    )
}




def train_detector_model(model):
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='loss', mode='min', min_delta=0.1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-accuracy{accuracy:.3f}_' + datetime.now().strftime(
                "%Y%m%d-%H%M%S") + '.h5',
            monitor='loss', mode='min')
    ]
    # -loss{loss:.3f}-val_loss{val_loss:.3f}_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer["adam"],
                  metrics=['accuracy'])

    train_dataset = generator_dataset_pair_creater(train_data)
    test_dataset = generator_dataset_pair_creater(test_data)
    logging.warning("Start train!")
    model.fit(train_dataset, batch_size=batch_size,
              epochs=epochs,
              callbacks=my_callbacks,
              validation_data=test_dataset)
