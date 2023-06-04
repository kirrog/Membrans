import tensorflow as tf
from keras import regularizers
from keras.layers import *
from keras.models import *

activation = 'relu'
padding = 'same'
n_filt = 16
noise = 0.3
p_drop = 0.4
kernel_initializer = 'he_normal'
channels = 3
pool_size = (2, 2)
deep = 3
kernel_regularizer_ = regularizers.L1L2(l1=1e-5, l2=1e-4)


def clearer_model_u_net(size=[512, 512]):
    img_rows, img_cols = size[0], size[1]
    input_shape = (img_rows, img_cols, 1)

    inputs = Input(input_shape)
    x = GaussianNoise(noise)(inputs)

    cross_data = []

    for i in range(deep - 1):
        x = Conv2D(n_filt * (pow(2, i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer_)(x)
        x = Conv2D(n_filt * (pow(2, i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer_)(x)
        x = BatchNormalization(axis=channels)(x)
        cross_data.append(x)
        x = MaxPooling2D(pool_size=pool_size)(x)
        x = Dropout(p_drop)(x)

    x = Conv2D(n_filt * 16, 1, activation=activation, padding=padding,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=kernel_regularizer_)(x)
    x = Conv2D(n_filt * 16, 1, activation=activation, padding=padding,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=kernel_regularizer_)(x)
    x = Dropout(p_drop)(x)

    for i in range(deep - 1):
        x = UpSampling2D(size=pool_size)(x)
        x = Conv2D(n_filt * (pow(2, deep - 2 - i)), 2, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer_)(x)
        merge = concatenate([cross_data.pop(), x], axis=channels)
        x = Conv2D(n_filt * (pow(2, deep - 2 - i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer_)(merge)
        x = Conv2D(n_filt * (pow(2, deep - 2 - i)), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer_)(x)
        if (i == deep - 2):
            x = Conv2D(2, 3, activation=activation, padding=padding,
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=kernel_regularizer_)(x)
        x = BatchNormalization(axis=channels)(x)

    x = Conv2D(1, 1, activation='sigmoid', kernel_regularizer=kernel_regularizer_)(x)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
    return model
