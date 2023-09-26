import keras.backend as K
import tensorflow as tf
from CapsuleKeras import *
from keras.models import Model
from keras import layers


def CapsNet(input_shape):
    x = layers.Input(shape=input_shape)

    conv1 = layers.Conv2D(filters=32, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    primarycap = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    digitcap = Capsule(num_capsule=10, dim_capsule=16, routings=3, name='digitcaps')(primarycap)

    out_caps = Length(name='capsnet')(digitcap)

    fc1 = layers.Dense(256, activation='relu')(out_caps)

    dp1 = layers.Dropout(0.5)(fc1)

    fc2 = layers.Dense(256, activation='relu')(dp1)

    dp2 = layers.Dropout(0.5)(fc2)

    output = layers.Dense(10, activation='softmax')(dp2)

    model = Model(inputs=x, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)
