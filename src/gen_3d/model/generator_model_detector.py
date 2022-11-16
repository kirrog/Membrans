from keras.layers import *
from keras.models import *

slice_height = 32


def detector_model_getter():
    a_regularizer = tf.keras.regularizers.L1L2(l1=0.00001, l2=0.0001)
    b_regularizer = tf.keras.regularizers.L1L2(l1=0.0001, l2=0.001)
    
    size = [slice_height, slice_height, slice_height]
    activation = 'relu'
    padding = 'same'
    n_filt = 16

    p_drop = 0.4
    kernel_initializer = 'he_normal'
    channels = 4
    pool_size = (2, 2, 2)

    num_of_slices_in_crater = 3

    img_rows, img_cols, deep_size = size[0], size[1], size[2]
    input_shape = (img_rows, img_cols, deep_size, 1)

    inputs = Input(input_shape)

    x = Conv3D(n_filt, 3, activation=activation, padding=padding,
               kernel_initializer=kernel_initializer, 
                   activity_regularizer=a_regularizer, bias_regularizer=b_regularizer)(inputs)

    for i in range(num_of_slices_in_crater, 0, -1):
        x = Conv3D(n_filt * pow(2, i), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer, 
                   activity_regularizer=a_regularizer, bias_regularizer=b_regularizer)(x)
        x = Conv3D(n_filt * pow(2, i), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer, 
                   activity_regularizer=a_regularizer, bias_regularizer=b_regularizer)(x)
        x = BatchNormalization(axis=channels)(x)
        x = MaxPooling3D(pool_size=pool_size)(x)
        x = Dropout(p_drop)(x)

    x = Conv3D(8, 8, activation=activation, padding=padding,
               kernel_initializer=kernel_initializer, 
                   activity_regularizer=a_regularizer, bias_regularizer=b_regularizer)(x)

    for i in range(num_of_slices_in_crater):
        x = UpSampling3D(size=pool_size)(x)
        x = Conv3D(n_filt * pow(2, i), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer, 
                   activity_regularizer=a_regularizer, bias_regularizer=b_regularizer)(x)
        x = Conv3D(n_filt * pow(2, i), 3, activation=activation, padding=padding,
                   kernel_initializer=kernel_initializer, 
                   activity_regularizer=a_regularizer, bias_regularizer=b_regularizer)(x)
        x = BatchNormalization(axis=channels)(x)
        x = Dropout(p_drop)(x)

    x = Dense(1)(x)

    model = Model(inputs=inputs, outputs=x)
    model.summary()

    return model
