from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf

#model_path = '/content/drive/MyDrive/Membrans/models/clearer/clearer_weights.h5'

model_path = '..\\models\\clearer\\clearer_weights.h5'

activation = 'relu'
padding = 'same'
n_filt = 64
noise = 0.3
p_drop = 0.4
kernel_initializer = 'he_normal'
channels = 3
pool_size = (2, 2)
deep = 5

ker_reg = tf.keras.regularizers.L1(0.01)
act_reg = tf.keras.regularizers.L2(0.01)

def clearer_model_new(size=[512, 512]):

    img_rows, img_cols = size[0], size[1]
    input_shape = (img_rows, img_cols, 1)

    inputs = Input(input_shape)
    x = GaussianNoise(noise)(inputs)

    cross_data = []

    for i in range(deep - 1):
        x = Conv2D(n_filt * (pow(2, i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer)(x)
        x = Conv2D(n_filt * (pow(2, i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization(axis=channels)(x)
        cross_data.append(x)
        x = MaxPooling2D(pool_size=pool_size)(x)
        x = Dropout(p_drop)(x)

    x = Conv2D(n_filt * 16, 1, activation=activation, padding=padding,
               kernel_initializer=kernel_initializer)(x)
    x = Conv2D(n_filt * 16, 1, activation=activation, padding=padding,
               kernel_initializer=kernel_initializer)(x)
    x = Dropout(p_drop)(x)

    for i in range(deep - 1):
        x = UpSampling2D(size=pool_size)(x)
        x = Conv2D(n_filt * (pow(2, deep - 2 - i)), 2, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer)(x)
        merge = concatenate([cross_data.pop(), x], axis=channels)
        x = Conv2D(n_filt * (pow(2, deep - 2 - i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer)(merge)
        x = Conv2D(n_filt * (pow(2, deep - 2 - i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer)(x)
        if (i == deep - 2):
            x = Conv2D(2, 3, activation=activation, padding=padding,
                       kernel_initializer=kernel_initializer)(x)
        x = BatchNormalization(axis=channels)(x)

    x = Conv2D(1, 1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


def clearer_model_new_resul(size=[512, 512]):
    n_filt = 40

    img_rows, img_cols = size[0], size[1]
    input_shape = (img_rows, img_cols, 1)

    inputs = Input(input_shape)
    x = GaussianNoise(noise)(inputs)

    cross_data = []

    for i in range(deep - 1):
        x = Conv2D(n_filt * (pow(2, i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
        x = Conv2D(n_filt * (pow(2, i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
        x = BatchNormalization(axis=channels, activity_regularizer=act_reg)(x)
        cross_data.append(x)
        x = MaxPooling2D(pool_size=pool_size)(x)
        x = Dropout(p_drop)(x)

    x = Conv2D(n_filt * 16, 1, activation=activation, padding=padding,
               kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
    x = Conv2D(n_filt * 16, 1, activation=activation, padding=padding,
               kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
    x = Dropout(p_drop)(x)

    for i in range(deep - 1):
        x = UpSampling2D(size=pool_size)(x)
        x = Conv2D(n_filt * (pow(2, deep - 2 - i)), 2, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
        merge = concatenate([cross_data.pop(), x], axis=channels)
        x = Conv2D(n_filt * (pow(2, deep - 2 - i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(merge)
        x = Conv2D(n_filt * (pow(2, deep - 2 - i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
        if (i == deep - 2):
            x = Conv2D(2, 3, activation=activation, padding=padding,
                       kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
        x = BatchNormalization(axis=channels, activity_regularizer=act_reg)(x)

    x = Conv2D(1, 1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


def clearer_model_new_deep(size=[512, 512]):
    n_filt = 8

    img_rows, img_cols = size[0], size[1]
    input_shape = (img_rows, img_cols, 1)

    inputs = Input(input_shape)
    x = GaussianNoise(noise)(inputs)

    deep = 2

    for i in range(deep - 1):
        x = Conv2D(n_filt * (pow(2, i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
        x = Conv2D(n_filt * (pow(2, i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
        x = BatchNormalization(axis=channels, activity_regularizer=act_reg)(x)
        x = MaxPooling2D(pool_size=pool_size)(x)
        x = Dropout(p_drop)(x)

    x = Conv2D(n_filt * 16, 1, activation=activation, padding=padding,
               kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
    x = Conv2D(n_filt * 16, 1, activation=activation, padding=padding,
               kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
    x = Dropout(p_drop)(x)

    for i in range(deep - 1):
        x = UpSampling2D(size=pool_size)(x)
        x = Conv2D(n_filt * (pow(2, deep - 2 - i)), 2, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
        x = Conv2D(n_filt * (pow(2, deep - 2 - i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
        x = Conv2D(n_filt * (pow(2, deep - 2 - i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
        if (i == deep - 2):
            x = Conv2D(2, 3, activation=activation, padding=padding,
                       kernel_initializer=kernel_initializer, activity_regularizer=act_reg)(x)
        x = BatchNormalization(axis=channels, activity_regularizer=act_reg)(x)

    x = Conv2D(1, 1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model


def load_model():
    new_model = keras.models.load_model(model_path)
    return new_model


def save_model(model):
    model.save(model_path)