#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 02:16:30 2017

@author: karim
"""



import math
import os
import os.path as osp
import sys
import time

import keras
import numpy as np
import tensorflow as tf


from datetime import datetime

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Activation, BatchNormalization, Convolution2D
from keras.layers import Dense, Dropout, Flatten, Input, LeakyReLU, Reshape
from keras.layers import UpSampling2D
from keras.optimizers import Adam
from PIL import Image


import arak
import arak.exp

from arak import TS, _DISP_SECTION_BREAKER_LEV0
from arak.util.path import makedirpath, splitroot


# ================================================================= Identifiers
# -----------------------------------------------------------------------------
_timestamp = datetime.now().strftime('%y%m%d%H%M%S%f')
_outDirPath = osp.join(os.getcwd(), 'tmp',
                       osp.split(splitroot(__file__)[-1])[0],
                       osp.splitext(osp.basename(__file__))[0], _timestamp)
makedirpath(_outDirPath)
# -----------------------------------------------------------------------------
# =============================================================================


# =============================================================================
# -----------------------------------------------------------------------------
IMG_ROWS, IMG_COLS = 28, 28
GENERATOR_CODE_DIM = 100
GENERATOR_OPTIMIZER = Adam(1e-4)
DISCRIMINATOR_OPTIMIZER = Adam(1e-3)
REPRESENTATION_DIMS = (IMG_ROWS, IMG_COLS, 1)
NUM_EPOCHS = 300
BATCH_SIZE = 100
NUM_GENERATORS = 2
# -----------------------------------------------------------------------------
# =============================================================================


def load_data():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape(trainX.shape[0], IMG_ROWS, IMG_COLS, 1)
    testX = testX.reshape(testX.shape[0], IMG_ROWS, IMG_COLS, 1)
    trainX = trainX.astype('float32') / 255.
    testX = testX.astype('float32') / 255.
    print TS('Loaded MNIST dataset')
    return (trainX, trainY), (testX, testY)


def set_trainable(model, trainable):
    model.trainable = trainable
    for x in model.layers:
        x.trainable = trainable
    return model


def freeze_model(model):
    return set_trainable(model, False)


def unfreeze_model(model):
    return set_trainable(model, True)


def is_frozen(model):
    return not model.trainable


def build_generator():
    print TS('Building the generator ...')
    inLayer = Input(shape=(GENERATOR_CODE_DIM,))
    H = Dense(100, init='he_normal')(inLayer)
    H = Activation('sigmoid')(H)
    H = Dense(7*7*128, init='he_normal')(inLayer)
    H = BatchNormalization(mode=2)(H)
    H = Activation('sigmoid')(H)
    H = Reshape([7, 7, 128])(H)
    H = UpSampling2D(size=(2, 2), dim_ordering='tf')(H)
    H = Convolution2D(nb_row=5,
                      nb_col=5,
                      nb_filter=64,
                      border_mode='same',
                      dim_ordering='tf',
                      init='he_normal')(H)
    H = BatchNormalization(mode=2)(H)
    H = Activation('sigmoid')(H)
    H = UpSampling2D(size=(2, 2), dim_ordering='tf')(H)
    H = Convolution2D(nb_row=5,
                      nb_col=5,
                      nb_filter=1,
                      border_mode='same',
                      dim_ordering='tf',
                      init='he_normal')(H)
    outLayer = Activation('sigmoid')(H)
    model = Model(input=inLayer, output=outLayer)
    model.compile(loss='binary_crossentropy',
                  optimizer=GENERATOR_OPTIMIZER)
    model.summary()
    return model


def build_discriminator():
    print TS('Building the discriminator ...')
    inLayer = Input(shape=REPRESENTATION_DIMS)
    H = Convolution2D(nb_filter=256,
                      nb_row=5,
                      nb_col=5,
                      subsample=(2, 2),
                      dim_ordering='tf',
                      border_mode='same')(inLayer)
    H = LeakyReLU(alpha=0.2)(H)
    H = Dropout(0.33)(H)
    H = Convolution2D(nb_filter=512,
                      nb_row=5,
                      nb_col=5,
                      subsample=(2, 2),
                      dim_ordering='tf',
                      border_mode='same')(H)
    H = LeakyReLU(alpha=0.2)(H)
    H = Dropout(0.33)(H)
    H = Flatten()(H)
    H = Dense(256)(H)
    H = LeakyReLU(alpha=0.2)(H)
    H = Dropout(0.33)(H)
    outLayer = Dense(1 + NUM_GENERATORS, activation='softmax')(H)
    model = Model(input=inLayer, output=outLayer)
    model.compile(loss='binary_crossentropy',
                  optimizer=DISCRIMINATOR_OPTIMIZER)
    model.summary()
    return model


def build_gan(gModel, dModel):
    print TS('Building the GAN model.')
    dModel = freeze_model(dModel)
    inLayer = Input(shape=(GENERATOR_CODE_DIM,))
    outLayer = dModel(gModel(inLayer))
    model = Model(input=inLayer, output=outLayer)
    model.compile(loss='categorical_crossentropy',
                  optimizer=GENERATOR_OPTIMIZER)
    model.summary()
    return model


def train_disc_on_batch(gModels, dModel, ganModel, inReal):
    st = time.time()
    inGEN = \
        np.random.uniform(0., 1., size=(inReal.shape[0], GENERATOR_CODE_DIM))
    inFake = gModels[0].predict(inGEN)
    refDISC = np.concatenate((np.zeros(inReal.shape[0]),
                              np.ones(inFake.shape[0])), axis=0)
    for x in range(1, len(gModels)):
        inFake = np.concatenate((inFake, gModels[x].predict(inGEN)), axis=0)
        refDISC = \
            np.concatenate((refDISC,
                            float(x + 1) * np.ones(inReal.shape[0])), axis=0)
    inDISC = np.concatenate((inFake, inReal), axis=0)
    refDISC = keras.utils.np_utils.to_categorical(refDISC, 1+len(gModels))
    assert(inFake.shape[0] == len(gModels) * inReal.shape[0])
    assert(np.array_equal(inFake.shape[1:], inReal.shape[1:]))
    assert(refDISC.shape[0] == inReal.shape[0] * (1 + len(gModels)))
    assert(refDISC.shape[1] == (1 + len(gModels)))
    assert(np.amin(refDISC) == 0 and np.amax(refDISC) == 1)
    assert(np.sum(refDISC) == refDISC.shape[0])
    assert(np.all(np.sum(refDISC, axis=1)))
    assert(np.array_equal(np.sum(refDISC, axis=0),
                          np.ones(len(gModels) + 1) * inReal.shape[0]))
    unfreeze_model(dModel)
    outLoss = dModel.train_on_batch(inDISC, refDISC)
    return outLoss, time.time() - st


def train_gen_on_batch(gModel, dModel, ganModel, num_samples, num_gens):
    st = time.time()
    inGAN = np.random.uniform(0., 1., size=(num_samples, GENERATOR_CODE_DIM))
    refGAN = np.zeros(num_samples)
    refGAN = keras.utils.np_utils.to_categorical(refGAN, num_gens+1)
    assert(np.array_equal(refGAN.shape, np.array([num_samples, num_gens+1])))
    assert(np.amin(refGAN) == 0 and np.amax(refGAN) == 1)
    freeze_model(dModel)
    outLoss = ganModel.train_on_batch(inGAN, refGAN)
    return outLoss, time.time() - st


def train_gan_on_batch(gModels, dModel, ganModels, inReal):
    dLosses, dTimes = [], []
    for x, y in zip(gModels, ganModels):
        dL, dT = train_disc_on_batch(gModels, dModel, ganModels, inReal)
        dLosses.append(dL)
        dTimes.append(dT)
    gLosses, gTimes = [], []
    for x, y in zip(gModels, ganModels):
        gL, gT = train_gen_on_batch(x, dModel, y,
                                    inReal.shape[0] * (len(gModels) + 1),
                                    len(gModels))
        gLosses.append(gL)
        gTimes.append(gT)
    return dLosses, gLosses, dTimes, gTimes


def generate_and_save(gModel, num_images, prefix):
    inGEN = np.random.uniform(0., 1., size=(num_images, GENERATOR_CODE_DIM))
    outGEN = np.squeeze(gModel.predict(inGEN))
    aggWidth = int(math.sqrt(num_images))
    aggHeight = int(math.ceil(float(num_images)/aggWidth))
    imgShape = outGEN.shape[1:]
    aggImg = np.zeros((aggHeight * imgShape[0],
                       aggWidth * imgShape[1]), dtype=outGEN.dtype)
    for index, img in enumerate(outGEN):
        i = int(index / aggWidth)
        j = index % aggWidth
        aggImg[i * imgShape[0]:(i+1) * imgShape[0],
               j * imgShape[1]:(j+1) * imgShape[1]] = img
    aggImg = aggImg * 255
    Image.fromarray(aggImg.astype(np.uint8)).save(
        osp.join(_outDirPath, '{}.png'.format(prefix)))


def fit_gan(gModels, dModel, ganModels, in_data, batch_size, num_epochs,
            shuffle=True):
    tmpInd = np.array(range(int(np.ceil(float(in_data.shape[0])/batch_size))))
    batch_start_ind = tmpInd * batch_size
    batch_end_ind = \
        np.clip((tmpInd + 1) * batch_size, a_min=None, a_max=in_data.shape[0])
    lossHistory = {'e': [], 'b': [], 'G': [], 'D': []}
    for eC in xrange(num_epochs):
        if shuffle:
            in_data = in_data[np.random.permutation(in_data.shape[0]), :]
        for bC in range(len(batch_start_ind)):
            step_time = time.time()
            bSI, bEI = batch_start_ind[bC], batch_end_ind[bC]
            dLoss, gLosses, dTime, gTimes = \
                train_gan_on_batch(gModels=gModels,
                                   dModel=dModel,
                                   ganModels=ganModels,
                                   inReal=in_data[bSI:bEI, :])
            lossHistory['e'].append(eC)
            lossHistory['b'].append(bC)
            lossHistory['G'].append(gLosses)
            lossHistory['D'].append(dLoss)
            st = time.time()
            if bC % 100 == 0:
                for x in range(len(gModels)):
                    generate_and_save(gModels[x], 1024,
                                      '{:02d}_{:03d}-{:03d}'.format(x, eC, bC))
            imgTime = time.time() - st
            print '... [{cS:0{fZ}d}/{tS}]) Step: {stepT:018.15f}'\
                .format(fZ=len(str(len(batch_start_ind))),
                        cS=bC, tS=len(batch_start_ind),
                        gL=gLosses, dL=dLoss, gT=gTimes, dT=dTime,
                        imgT=imgTime, stepT=time.time()-step_time)
        print 'E {cE:0{fZ}}/{tE}) /*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/'\
            .format(fZ=len(str(num_epochs)),
                    cE=eC, tE=num_epochs, gL=gLosses, dL=dLoss)


if __name__ == '__main__':
    np.random.seed(123)
    (trainX, trainY), (testX, testY) = load_data()
    GENs = [build_generator() for x in range(NUM_GENERATORS)]
    DSC = build_discriminator()
    GANs = [build_gan(x, DSC) for x in GENs]
    fit_gan(gModels=GENs,
            dModel=DSC,
            ganModels=GANs,
            in_data=trainX,
            batch_size=BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            shuffle=True)
