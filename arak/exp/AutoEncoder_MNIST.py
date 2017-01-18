#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 09:53:14 2017

@author: karim
"""

import os
import os.path as osp
import sys

import numpy as np

from datetime import datetime

from keras.layers import Activation, Dense, Input
from keras.layers import LeakyReLU
from keras.models import Model, Sequential
from keras.optimizers import Adam
from sklearn.datasets import fetch_mldata



import arak
import arak.exp

from arak.util.path import makedirpath, splitroot


# ================================================================= Identifiers
_timestamp = datetime.now().strftime('%y%m%d%H%M%S%f')
_outDirPath = osp.join(os.getcwd(), 'tmp',
                       osp.split(splitroot(__file__)[-1])[0],
                       osp.splitext(osp.basename(__file__))[0], _timestamp)
makedirpath(_outDirPath)
# =============================================================================


def get_dataset():
    dataset = fetch_mldata('MNIST Original')
    return dataset.data


def build_encoder(in_dim, encoding_dims,
                  hidden_activation=lambda: Activation('relu')):
    encoder_input_layer = Input(shape=(in_dim,))
    layer = encoder_input_layer
    for x in encoding_dims:
        layer = hidden_activation()(Dense(x)(layer))
    encoder = Model(input=encoder_input_layer, output=layer)
    return encoder


def build_decoder(in_dim, decoding_dims, out_dim,
                  hidden_activation=lambda: Activation('relu'),
                  output_activation=lambda: Activation('sigmoid')):
    decoder_input_layer = Input(shape=(in_dim,))
    layer = decoder_input_layer
    for x in decoding_dims:
        layer = hidden_activation()(Dense(x)(layer))
    layer = output_activation()(Dense(out_dim)(layer))
    decoder = Model(input=decoder_input_layer, output=layer)
    return decoder


def build_autoencoder(in_dim=784, encoding_dims=[32],
                      decoding_dims=[], out_dim=784,
                      encoder_hidden_activation=lambda: Activation('relu'),
                      decoder_hidden_activation=lambda: Activation('relu'),
                      decoder_output_activation=lambda: Activation('sigmoid')):
    input_layer = Input(shape=(in_dim,))
    encoder = build_encoder(in_dim, encoding_dims,
                            hidden_activation=encoder_hidden_activation)
    decoder = build_decoder(encoding_dims[-1], decoding_dims, out_dim,
                            hidden_activation=decoder_hidden_activation,
                            output_activation=decoder_output_activation)
    autoencoder = Model(input=input_layer,
                        output=decoder(encoder(input_layer)))
    return autoencoder, encoder, decoder


if __name__ == '__main__':
    trainX = get_dataset().astype(np.float32) / 255.
    testX = trainX[:10000, :]
    trainX = trainX[10000:, :]

    AE, encoder, decoder = \
        build_autoencoder(
            encoder_hidden_activation=lambda: LeakyReLU(alpha=0.1),
            decoder_hidden_activation=lambda: LeakyReLU(alpha=0.1),
            decoder_output_activation=lambda: Activation('sigmoid'))
    AE.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy')
    
    encoder.summary()
    decoder.summary()
    AE.summary()

    AE.fit(trainX, trainX,
           nb_epoch=100,
           batch_size=256,
           shuffle=True,
           validation_data=(testX, testX))
