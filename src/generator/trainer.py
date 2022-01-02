import tensorflow as tf
from datetime import datetime

# from clearer.datasets_loader import load_clearer_dataset_predicts, load_clearer_dataset_answers
# from utils.augmentations import augment_dataset
# from utils.augment_dataset_generator import augment_size, get_augment_dataset
from src.generator.dataset_generator_providers_old import generator_dataset_pair_creater
from src.generator.datasets_loader import load_generator_dataset_predicts, load_generator_dataset_answers
from src.utils.augmentations import augment_dataset
from src.utils.augment_dataset_generator import augment_size, get_augment_dataset

# log_dir = '/content/drive/MyDrive/Membrans/models/generator/logs/'

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

    train_dataset, test_dataset = generator_dataset_pair_creater()

    model.fit(train_dataset, batch_size=batch_size,
              epochs=epochs,
              callbacks=my_callbacks,
              validation_data=test_dataset)
