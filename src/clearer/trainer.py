import tensorflow as tf
import time
from datetime import datetime
from src.clearer.datasets_loader import load_clearer_dataset

def train_clearer_model(model):
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='loss', mode='min', min_delta=0),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='model.{epoch:02d}-{val_dice2:.4f}-{val_loss:.6f}' + datetime.now().strftime(
                "%Y%m%d-%H%M%S") + '.h5',
            monitor='val_dice_coef', verbose=1,
            save_best_only=True, mode='max')
    ]

    predictors, answers = load_clearer_dataset()

    batch_size = 6
    epochs = 500

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    start = time.time()

    history = model.fit(predictors, answers, batch_size=batch_size,
                        epochs=epochs, verbose=1,
                        validation_split=0.1,
                        callbacks=my_callbacks)

    end = time.time()
    print('Time of learning: ', end - start)
    print(history)

