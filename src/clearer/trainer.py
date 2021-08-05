import tensorflow as tf
import time
from datetime import datetime
from src.clearer.datasets_loader import load_clearer_dataset

log_dir = '..\\models\\clearer\\'

def train_clearer_model(model):
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='loss', mode='min', min_delta=0),
        tf.keras.callbacks.ModelCheckpoint(filepath=log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}_'+
                                           datetime.now().strftime("%Y%m%d-%H%M%S") + '.h5',
                        monitor='val_loss', save_weights_only=True, save_best_only=True, mode='max')
    ]

    predictors, answers = load_clearer_dataset()

    batch_size = 6
    epochs = 500

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(predictors, answers, batch_size=batch_size,
                        epochs=epochs, verbose=1,
                        validation_split=0.1,
                        callbacks=my_callbacks)

    logs = open('..\\logs\\' + (datetime.now().strftime("%Y-%m-%d-%H.%M.%S") ), 'w')
    logs.write(str(history))
    logs.close()
