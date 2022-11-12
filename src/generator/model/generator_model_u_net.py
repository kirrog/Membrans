from keras.layers import *
from keras.layers import Resizing
from keras.models import *
from tensorflow import keras

model_path = '../models/generator/generator_weights.h5'


def get_attention_on_tooth(y, cross_data):
    y = y[:, :32, :]
    y = Dense(units=32, activation='relu')(y)
    y = Reshape((32, 32, 1))(y)
    for i in range(len(cross_data)):
        x = cross_data[i]
        x = Resizing(height=32, width=32)(x)
        y = Conv2D(32 * pow(2, i), 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(y)
        y = Multiply()([x, y])
    return y


def generator_model_u_net():
    activation = 'relu'
    padding = 'same'
    n_filt = 32
    noise = 0.3
    p_drop = 0.4
    kernel_initializer = 'he_normal'
    channels = 3
    pool_size = (2, 2)

    num_of_slices_in_crater = 4

    img_rows, img_cols = 512, 512
    input_shape = (img_rows, img_cols, 1)

    inputs = Input(input_shape)
    # inputs_y = inputs[:, img_rows - 1, :, :]
    # inputs_x = inputs[:, :img_rows - 1, :, :]

    x = GaussianNoise(noise)(inputs)  # may be deleted

    # cross_data = []

    for i in range(num_of_slices_in_crater):
        x = Conv2D(n_filt * pow(2, i), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer)(x)
        x = Conv2D(n_filt * pow(2, i), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization(axis=channels)(x)
        # cross_data.append(x)
        x = MaxPooling2D(pool_size=pool_size)(x)
        x = Dropout(p_drop)(x)

    # inputs_y = get_attention_on_tooth(inputs_y, cross_data)

    # x = Multiply()([x, inputs_y])
    x = Conv2D(n_filt * pow(2, num_of_slices_in_crater), 1, activation=activation, padding=padding,
               kernel_initializer=kernel_initializer)(x)
    x = Dropout(p_drop)(x)

    for i in range(num_of_slices_in_crater):
        x = UpSampling2D(size=pool_size)(x)
        x = Conv2D(n_filt * pow(2, num_of_slices_in_crater - 1 - i), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer)(x)
        x = Conv2D(n_filt * pow(2, num_of_slices_in_crater - 1 - i), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization(axis=channels)(x)

    # x = Conv2D(2, 1, activation='sigmoid')(x)
    x = Conv2D(1, 1, activation='sigmoid')(x)

    return Model(inputs=inputs, outputs=x)


def load_model():
    new_model = keras.models.load_model(model_path)
    return new_model


def save_generator_model(model):
    model.save(model_path)
