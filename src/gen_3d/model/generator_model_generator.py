from keras.layers import *
from keras.models import *


def generator_model_coder(size=[512, 512, 1]):
    activation = 'relu'
    padding = 'same'
    n_filt = 16

    p_drop = 0.4
    kernel_initializer = 'he_normal'
    channels = 3
    pool_size = (2, 2)

    num_of_slices_in_crater = 4

    img_rows, img_cols = size[0], size[1]
    input_shape = (img_rows, img_cols, 1)

    inputs = Input(input_shape)

    cross_data = []

    x = Conv2D(n_filt, 3, activation=activation, padding=padding,
               kernel_initializer=kernel_initializer)(inputs)

    for i in range(num_of_slices_in_crater):
        x = Conv2D(n_filt * pow(2, i), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer)(x)
        x = Conv2D(n_filt * pow(2, i), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization(axis=channels)(x)
        cross_data.append(x)
        x = MaxPooling2D(pool_size=pool_size)(x)
        x = Dropout(p_drop)(x)

    x = Conv2D(n_filt * pow(2, num_of_slices_in_crater), 1, activation=activation, padding=padding,
               kernel_initializer=kernel_initializer)(x)
    x = Dropout(p_drop)(x)

    model = Model(inputs=inputs, outputs=x)
    print(model.output_shape)
    assert model.output_shape == (None, 32, 32, 256)
    return model


def generator_model_decoder(size=[32, 32, 256]):
    activation = 'relu'
    padding = 'same'
    n_filt = 16
    p_drop = 0.4
    kernel_initializer = 'he_normal'
    channels = 3
    pool_size = (2, 2)
    noise = 0.3

    num_of_slices_in_crater = 4

    img_rows, img_cols = size[0], size[1]
    input_shape = (img_rows, img_cols, 256)

    inputs = Input(input_shape)
    x = GaussianNoise(noise)(inputs)

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

    x = Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    print(model.output_shape)
    assert model.output_shape == (None, 512, 512, 1)
    return model


def generator_model_discriminator(size=[512, 512, 1]):
    img_rows, img_cols = size[0], size[1]
    input_shape = (img_rows, img_cols, 1)

    inputs = Input(input_shape)
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(inputs)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(1)(x)

    model = Model(inputs=inputs, outputs=x)
    print(model.output_shape)
    assert model.output_shape == (None, 1)
    return model
