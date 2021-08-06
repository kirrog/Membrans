from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

model_path = '..\\models\\clearer\\clearer_weights.h5'


def clearer_model_new(size=[512, 512], n_filt = 64):
    activation = 'relu'
    padding = 'same'
    noise = 0.3
    p_drop = 0.4
    kernel_initializer = 'he_normal'
    channels = 3
    pool_size = (2, 2)

    img_rows, img_cols = size[0], size[1]
    input_shape = (img_rows, img_cols, 1)

    inputs = Input(input_shape)
    inp_noise = GaussianNoise(noise)(inputs)

    conv1 = Conv2D(n_filt, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(inp_noise)
    conv1 = Conv2D(n_filt, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv1)
    batch1 = BatchNormalization(axis=channels)(conv1)
    pool1 = MaxPooling2D(pool_size=pool_size)(batch1)
    drop1 = Dropout(p_drop)(pool1)

    conv2 = Conv2D(n_filt * 2, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(drop1)
    conv2 = Conv2D(n_filt * 2, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv2)
    batch2 = BatchNormalization(axis=channels)(conv2)
    pool2 = MaxPooling2D(pool_size=pool_size)(batch2)
    drop2 = Dropout(p_drop)(pool2)

    conv3 = Conv2D(n_filt * 4, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(drop2)
    conv3 = Conv2D(n_filt * 4, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv3)
    batch3 = BatchNormalization(axis=channels)(conv3)
    pool3 = MaxPooling2D(pool_size=pool_size)(batch3)
    drop3 = Dropout(p_drop)(pool3)

    conv4 = Conv2D(n_filt * 8, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(drop3)
    conv4 = Conv2D(n_filt * 8, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv4)
    batch4 = BatchNormalization(axis=channels)(conv4)
    pool4 = MaxPooling2D(pool_size=pool_size)(batch4)
    drop4 = Dropout(p_drop)(pool4)

    conv5 = Conv2D(n_filt * 16, 1, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(drop4)
    conv5 = Conv2D(n_filt * 16, 1, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv5)
    drop5 = Dropout(p_drop)(conv5)

    up6 = UpSampling2D(size=pool_size)(drop5)
    up6 = Conv2D(n_filt * 8, 2, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(up6)
    merge6 = concatenate([batch4, up6], axis=channels)
    conv6 = Conv2D(n_filt * 8, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(merge6)
    conv6 = Conv2D(n_filt * 8, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv6)
    batch6 = BatchNormalization(axis=channels)(conv6)

    up7 = UpSampling2D(size=pool_size)(batch6)
    up7 = Conv2D(n_filt * 4, 2, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(up7)
    merge7 = concatenate([batch3, up7], axis=channels)
    conv7 = Conv2D(n_filt * 4, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(merge7)
    conv7 = Conv2D(n_filt * 4, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv7)
    batch7 = BatchNormalization(axis=channels)(conv7)

    up8 = UpSampling2D(size=pool_size)(batch7)
    up8 = Conv2D(n_filt * 2, 2, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(up8)
    merge8 = concatenate([batch2, up8], axis=channels)

    conv8 = Conv2D(n_filt * 2, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(merge8)
    conv8 = Conv2D(n_filt * 2, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv8)
    batch8 = BatchNormalization(axis=channels)(conv8)

    up9 = UpSampling2D(size=pool_size)(batch8)
    up9 = Conv2D(n_filt, 2, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(up9)
    merge9 = concatenate([batch1, up9], axis=channels)
    conv9 = Conv2D(n_filt, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(merge9)
    conv9 = Conv2D(n_filt, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv9)
    conv9 = Conv2D(2, 3, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv9)
    batch9 = BatchNormalization(axis=channels)(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid')(batch9)
    return Model(inputs=inputs, outputs=conv10)


def load_model():
    new_model = keras.models.load_model(model_path)
    return new_model


def save_model(model):
    model.save(model_path)