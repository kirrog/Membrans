from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow import keras

model_path = '..\\models\\generator\\generator_weights.h5'

def generator_model_new(size=[512, 512]):
    activation = 'relu'
    padding = 'same'
    n_filt = 64
    noise = 0.3
    p_drop = 0.4
    kernel_initializer = 'he_normal'
    channels = 3
    pool_size = (2, 2)

    num_of_slices_in_crater = 4

    img_rows, img_cols = size[0], size[1]
    input_shape = (img_rows, img_cols, 1)

    inputs = Input(input_shape)
    x = GaussianNoise(noise)(inputs)

    for i in range(num_of_slices_in_crater):
        x = Conv2D(n_filt * pow(2, i), 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(x)
        x = Conv2D(n_filt * pow(2, i), 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization(axis=channels)(x)
        x = MaxPooling2D(pool_size=pool_size)(x)
        x = Dropout(p_drop)(x)

    x = Conv2D(n_filt * 16, 1, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = Conv2D(n_filt * 16, 1, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = Dropout(p_drop)(x)

    for i in range(num_of_slices_in_crater):
        x = UpSampling2D(size=pool_size)(x)
        x = Conv2D(n_filt * pow(2, num_of_slices_in_crater-1-i), 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(x)
        x = Conv2D(n_filt * pow(2, num_of_slices_in_crater-1-i), 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization(axis=channels)(x)

    x = Conv2D(1, 1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=x)


def load_model():
    new_model = keras.models.load_model(model_path)
    return new_model


def save_generator_model(model):
    model.save(model_path)

