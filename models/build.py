#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 18/7/31 10:06
# @Author   : Yiwen Liao
# @File     : visualizations.py
# @Software : PyCharm
# @License  : Copyright(C), Yiwen Liao
# @Contact  : yiwen.liao93@gmail.com


from useful_packages import *


# ==================== ICSNET ====================
def build_icsnet(img_shape=None, reg=None, latent_fea=None):
    """Create ICSNET model.

    :param img_shape: A tuple standing for image shape, i.e. (height, width, # channels).
    :param reg: Float value for weight decay.
    :param latent_fea: A desired integer number of latent features.
    :return: An ICSNET model and a joint-ICSNET model.
    """
    # ==================== Constants Definition ====================
    acti_func = 'linear'
    clf_acti = 'sigmoid'

    acti_alpha = 0.2
    set_bias = False

    weights_init = tn(mean=0, stddev=0.01)

    bn_eps = 1e-3
    bn_m = 0.99

    # ==================== ICSNET ====================

    input_layer = Input(shape=img_shape, name='input_layer')

    conv_1 = Conv2D(filters=16, kernel_size=(3, 3), activation=acti_func, name='conv_1',
                    padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                    kernel_initializer=weights_init)(input_layer)
    conv_11 = Conv2D(filters=16, kernel_size=(3, 3), activation=acti_func, name='conv_11',
                     padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                     kernel_initializer=weights_init)(conv_1)
    conv_1 = Concatenate()([conv_1, conv_11])  # 32x32x64
    lrelu_1 = LeakyReLU(alpha=acti_alpha)(conv_1)
    pool_1 = AveragePooling2D(pool_size=(2, 2), name='pool_1')(lrelu_1)  # 16x16 / 14x14
    bn_1 = BatchNormalization(momentum=bn_m, epsilon=bn_eps, name='bn_1')(pool_1)

    conv_2 = Conv2D(filters=32, kernel_size=(3, 3), activation=acti_func, name='conv_2',
                    padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                    kernel_initializer=weights_init)(bn_1)
    conv_22 = Conv2D(filters=32, kernel_size=(3, 3), activation=acti_func, name='conv_22',
                     padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                     kernel_initializer=weights_init)(conv_2)
    conv_2 = Concatenate()([conv_2, conv_22])  # 16x16x128
    lrelu_2 = LeakyReLU(alpha=acti_alpha)(conv_2)
    pool_2 = AveragePooling2D(pool_size=(2, 2), name='pool_2')(lrelu_2)  # 8x8 / 7x7
    if img_shape[0] == 28:
        pool_2 = ZeroPadding2D(padding=(1, 1))(pool_2)  # zero-padding if mnist or fashion-mnist
    bn_2 = BatchNormalization(momentum=bn_m, epsilon=bn_eps, name='bn_2')(pool_2)

    conv_3 = Conv2D(filters=64, kernel_size=(3, 3), activation=acti_func, name='conv_3',
                    padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                    kernel_initializer=weights_init)(bn_2)
    conv_33 = Conv2D(filters=64, kernel_size=(3, 3), activation=acti_func, name='conv_33',
                     padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                     kernel_initializer=weights_init)(conv_3)
    conv_3 = Concatenate()([conv_3, conv_33])  # 8x8x256
    lrelu_3 = LeakyReLU(alpha=acti_alpha)(conv_3)
    pool_3 = AveragePooling2D(pool_size=(2, 2), name='pool_3')(lrelu_3)  # 4x4
    bn_3 = BatchNormalization(momentum=bn_m, epsilon=bn_eps, name='bn_3')(pool_3)

    conv_4 = Conv2D(filters=128, kernel_size=(3, 3), activation=acti_func, name='conv_4',
                    padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                    kernel_initializer=weights_init)(bn_3)
    conv_44 = Conv2D(filters=128, kernel_size=(3, 3), activation=acti_func, name='conv_44',
                     padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                     kernel_initializer=weights_init)(conv_4)
    conv_4 = Concatenate()([conv_4, conv_44])  # 4x4x512
    lrelu_4 = LeakyReLU(alpha=acti_alpha)(conv_4)
    pool_4 = AveragePooling2D(pool_size=(2, 2), name='pool_4')(lrelu_4)  # 2x2
    bn_4 = BatchNormalization(momentum=bn_m, epsilon=bn_eps, name='bn_4')(pool_4)

    conv_5 = Conv2D(filters=256, kernel_size=(1, 1), activation=acti_func, name='conv_5',
                    kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                    kernel_initializer=weights_init)(bn_4)  # 2x2
    conv_55 = Conv2D(filters=256, kernel_size=(1, 1), activation=acti_func, name='conv_55',
                     kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                     kernel_initializer=weights_init)(bn_4)
    conv_5 = Concatenate()([conv_5, conv_55])  # 2x2x512
    lrelu_5 = LeakyReLU(alpha=acti_alpha)(conv_5)
    bn_5 = BatchNormalization(name='bn_5')(lrelu_5)

    flt_6 = Flatten()(bn_5)

    dense_7 = Dense(units=256, activation=acti_func, name='dense_7',
                    kernel_regularizer=regularizers.l2(reg), use_bias=not set_bias,
                    kernel_initializer=weights_init)(flt_6)
    lrelu_7 = LeakyReLU(alpha=acti_alpha)(dense_7)
    drop_7 = Dropout(rate=0.5)(lrelu_7)

    dense_8 = Dense(units=latent_fea, activation=acti_func, name='dense_8',
                    kernel_regularizer=regularizers.l2(reg), use_bias=not set_bias,
                    kernel_initializer=weights_init)(drop_7)

    lrelu_8 = LeakyReLU(alpha=acti_alpha)(dense_8)
    drop_8 = Dropout(rate=0.5)(lrelu_8)

    dense_9 = Dense(units=1, activation='sigmoid', name='dense_9',
                    kernel_regularizer=regularizers.l2(reg), use_bias=not set_bias,
                    kernel_initializer=weights_init)(drop_8)

    output_layer = Reshape(target_shape=(-1,), name='top_layer')(dense_9)
    latent_layer = Reshape(target_shape=(-1,), name='latent_layer')(lrelu_8)

    icsnet = Model(inputs=input_layer, outputs=output_layer, name='ICSNET')
    icsnet_latent = Model(inputs=input_layer, outputs=latent_layer, name='ICSNET_latent')

    # ==================== Subnetwork ====================

    input_1 = Input(shape=img_shape, name='input_1')
    input_2 = Input(shape=img_shape, name='input_2')

    lat_1 = icsnet_latent(input_1)
    lat_2 = icsnet_latent(input_2)

    latent_dist = Subtract(name='latent_dist')([lat_1, lat_2])

    dense_ly = Dense(units=1, activation=clf_acti, name='dense_ly',
                     kernel_regularizer=regularizers.l2(reg))(latent_dist)

    ic_network = Model(inputs=[input_1, input_2], outputs=dense_ly)

    # ==================== Multiple Input Layers ====================

    typical_input_1 = Input(shape=img_shape, name='typical_input_1')
    typical_input_2 = Input(shape=img_shape, name='typical_input_2')
    atypical_input_1 = Input(shape=img_shape, name='atypical_input_1')
    atypical_input_2 = Input(shape=img_shape, name='atypical_input_2')

    typical_dist = ic_network([typical_input_1, typical_input_2])
    atypical_dist = ic_network([atypical_input_1, atypical_input_2])

    # ==================== Final Model ====================

    icsnet_joint = Model(inputs=[typical_input_1, typical_input_2, atypical_input_1, atypical_input_2],
                         outputs=[typical_dist, atypical_dist], name='ICSNET_joint')
    icsnet.summary()

    return icsnet, icsnet_joint


# ==================== Autoencoders ====================
def build_ae(img_shape=None, reg=None, latent_fea=None):
    """Create autoencoder models.

    :param img_shape: A tuple standing for image shape, i.e. (height, width, # channels).
    :param reg: Float value for weight decay.
    :param latent_fea: A desired integer number of latent features.
    :return: An autoencoder model.
    """
    # ==================== Constants Definition ====================
    acti_func = 'relu'
    set_bias = False
    weights_init = tn(mean=0, stddev=0.01)

    # ==================== Encoder ====================

    input_layer = Input(shape=img_shape, name='input_layer')

    if img_shape[0] == 28:
        pad_layer = ZeroPadding2D(padding=(2, 2))(input_layer)
    else:
        pad_layer = input_layer

    conv_1 = Conv2D(filters=8, kernel_size=(3, 3), activation=acti_func, name='conv_1',
                    kernel_initializer=weights_init, padding='same',
                    kernel_regularizer=regularizers.l2(reg), use_bias=set_bias)(pad_layer)  # 32x32
    conv_11 = Conv2D(filters=8, kernel_size=(3, 3), activation=acti_func, name='conv_11',
                     kernel_initializer=weights_init, padding='same',
                     kernel_regularizer=regularizers.l2(reg), use_bias=set_bias)(conv_1)
    conv_1 = Concatenate()([conv_11, conv_1])

    pool_1 = MaxPooling2D(pool_size=(2, 2), name='pool_1')(conv_1)  # 16x16

    bn_1 = BatchNormalization(name='bn_1')(pool_1)

    conv_2 = Conv2D(filters=16, kernel_size=(3, 3), activation=acti_func, name='conv_2',
                    kernel_initializer=weights_init, padding='same',
                    kernel_regularizer=regularizers.l2(reg), use_bias=set_bias)(bn_1)
    conv_22 = Conv2D(filters=16, kernel_size=(3, 3), activation=acti_func, name='conv_22',
                     kernel_initializer=weights_init, padding='same',
                     kernel_regularizer=regularizers.l2(reg), use_bias=set_bias)(conv_2)
    conv_2 = Concatenate()([conv_22, conv_2])

    pool_2 = MaxPooling2D(pool_size=(2, 2), name='pool_2')(conv_2)  # 8x8

    bn_2 = BatchNormalization(name='bn_2')(pool_2)

    conv_3 = Conv2D(filters=32, kernel_size=(3, 3), activation=acti_func, name='conv_3',
                    kernel_initializer=weights_init, padding='same',
                    kernel_regularizer=regularizers.l2(reg), use_bias=set_bias)(bn_2)
    conv_33 = Conv2D(filters=32, kernel_size=(3, 3), activation=acti_func, name='conv_33',
                     kernel_initializer=weights_init, padding='same',
                     kernel_regularizer=regularizers.l2(reg), use_bias=set_bias)(conv_3)
    conv_3 = Concatenate()([conv_33, conv_3])

    pool_3 = MaxPooling2D(pool_size=(2, 2), name='pool_3')(conv_3)  # 4x4

    bn_3 = BatchNormalization(name='bn_3')(pool_3)

    conv_4 = Conv2D(filters=64, kernel_size=(3, 3), activation=acti_func, name='conv_4',
                    kernel_regularizer=regularizers.l2(reg), padding='same',
                    kernel_initializer=weights_init, use_bias=set_bias)(bn_3)
    conv_44 = Conv2D(filters=64, kernel_size=(3, 3), activation=acti_func, name='conv_44',
                     kernel_regularizer=regularizers.l2(reg), padding='same',
                     kernel_initializer=weights_init, use_bias=set_bias)(conv_4)
    conv_4 = Concatenate()([conv_44, conv_4])

    pool_4 = MaxPooling2D(pool_size=(2, 2), name='pool_4')(conv_4)  # 2x2

    bn_4 = BatchNormalization(name='bn_4')(pool_4)

    conv_lat = Conv2D(filters=latent_fea, kernel_size=(1, 1), activation=acti_func, name='conv_lat',
                      kernel_regularizer=regularizers.l2(reg), padding='same',
                      kernel_initializer=weights_init, use_bias=not set_bias)(bn_4)

    # ==================== Decoder ====================

    bn_5 = BatchNormalization(name='bn_lat')(conv_lat)

    convt_5 = Conv2DTranspose(filters=32, kernel_size=(3, 3), activation=acti_func, name='convt_5',
                              kernel_initializer=weights_init, padding='same', strides=(2, 2),
                              kernel_regularizer=regularizers.l2(reg), use_bias=set_bias)(bn_5)
    convt_55 = Conv2D(filters=32, kernel_size=(3, 3), activation=acti_func, name='convt_55',
                      kernel_regularizer=regularizers.l2(reg), padding='same',
                      kernel_initializer=weights_init, use_bias=set_bias)(convt_5)
    convt_5 = Concatenate()([convt_55, convt_5])

    bn_5 = BatchNormalization(name='bn_5')(convt_5)  # 4x4

    convt_6 = Conv2DTranspose(filters=32, kernel_size=(3, 3), activation=acti_func, name='convt_6',
                              padding='same', kernel_initializer=weights_init, strides=(2, 2),
                              kernel_regularizer=regularizers.l2(reg), use_bias=set_bias)(bn_5)
    convt_66 = Conv2D(filters=32, kernel_size=(3, 3), activation=acti_func, name='convt_66',
                      kernel_regularizer=regularizers.l2(reg), padding='same',
                      kernel_initializer=weights_init, use_bias=set_bias)(convt_6)
    convt_6 = Concatenate()([convt_66, convt_6])

    bn_6 = BatchNormalization(name='bn_6')(convt_6)  # 8x8

    convt_7 = Conv2DTranspose(filters=16, kernel_size=(3, 3), activation=acti_func, name='convt_7',
                              padding='same', kernel_initializer=weights_init, strides=(2, 2),
                              kernel_regularizer=regularizers.l2(reg), use_bias=set_bias)(bn_6)
    convt_77 = Conv2D(filters=16, kernel_size=(3, 3), activation=acti_func, name='convt_77',
                      kernel_regularizer=regularizers.l2(reg), padding='same',
                      kernel_initializer=weights_init, use_bias=set_bias)(convt_7)
    convt_7 = Concatenate()([convt_77, convt_7])

    bn_7 = BatchNormalization(name='bn_7')(convt_7)  # 16x16

    convt_8 = Conv2DTranspose(filters=8, kernel_size=(3, 3), activation=acti_func, name='convt_8',
                              padding='same', kernel_initializer=weights_init, strides=(2, 2),
                              kernel_regularizer=regularizers.l2(reg), use_bias=set_bias)(bn_7)
    convt_88 = Conv2D(filters=8, kernel_size=(3, 3), activation=acti_func, name='convt_88',
                      kernel_regularizer=regularizers.l2(reg), padding='same',
                      kernel_initializer=weights_init, use_bias=set_bias)(convt_8)
    convt_8 = Concatenate()([convt_88, convt_8])

    bn_8 = BatchNormalization(name='bn_8')(convt_8)  # 32x32

    conv_9 = Conv2D(filters=img_shape[-1], kernel_size=(3, 3), activation='sigmoid', name='conv_9',
                    kernel_initializer=weights_init, padding='same',
                    kernel_regularizer=regularizers.l2(reg), use_bias=not set_bias)(bn_8)

    if img_shape[0] == 28:
        conv_9 = Cropping2D(cropping=((2, 2), (2, 2)))(conv_9)
    else:
        pass

    ae = Model(inputs=input_layer, outputs=conv_9, name='autoencoder')
    ae.summary()

    return ae
