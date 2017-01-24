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
    H = BatchNormalization(H)
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
    outLayer = Dense(2, activation='softmax')(H)
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


def train_disc_on_batch(gModel, dModel, ganModel, inReal):
    st = time.time()
    inGEN = \
        np.random.uniform(0., 1., size=(inReal.shape[0], GENERATOR_CODE_DIM))
    inFake = gModel.predict(inGEN)
    inDISC = np.concatenate((inFake, inReal), axis=0)
    refDISC = np.concatenate((np.zeros(inFake.shape[0]),
                              np.ones(inReal.shape[0])), axis=0)
    refDISC = keras.utils.np_utils.to_categorical(refDISC, 2)
    assert(np.array_equal(inFake.shape, inReal.shape))
    assert(np.array_equal(refDISC.shape, np.array([inReal.shape[0]*2, 2])))
    assert(np.amin(refDISC) == 0 and np.amax(refDISC) == 1)
    assert(is_frozen(dModel))
    unfreeze_model(dModel)
    outLoss = dModel.train_on_batch(inDISC, refDISC)
    outDISC = dModel.predict(inDISC)
    return outLoss, time.time() -st


def train_gen_on_batch(gModel, dModel, ganModel, batch_size):
    st = time.time()
    inGAN = \
        np.random.uniform(0., 1., size=(2 * batch_size, GENERATOR_CODE_DIM))
    refGAN = np.ones(2 * batch_size)
    refGAN = keras.utils.np_utils.to_categorical(refGAN, 2)
    assert(np.array_equal(refGAN.shape, np.array([2 * batch_size, 2])))
    assert(np.amin(refGAN) == 0 and np.amax(refGAN) == 1)
    assert(not is_frozen(dModel))
    freeze_model(dModel)
    outLoss = ganModel.train_on_batch(inGAN, refGAN)
    return outLoss, time.time() -st


def train_gan_on_batch(gModel, dModel, ganModel, inReal):
    dLoss, dTime = train_disc_on_batch(gModel, dModel, ganModel, inReal)
    gLoss, gTime = train_gen_on_batch(gModel, dModel, ganModel, inReal.shape[0])
    return dLoss, gLoss, dTime, gTime


def generate_and_save(gModel, dModel, ganModel, num_images, prefix):
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


def fit_gan(gModel, dModel, ganModel, in_data, batch_size, num_epochs,
            shuffle=True):
    tmpInd = np.array(range(int(np.ceil(float(in_data.shape[0])/batch_size))))
    batch_start_ind = tmpInd * batch_size
    batch_end_ind = \
        np.clip((tmpInd + 1) * batch_size, a_min=None, a_max=in_data.shape[0])
    lossHistory = {'e': [], 'b': [], 'G': [], 'D': []}
    for eCounter in xrange(num_epochs):
        if shuffle:
            in_data = in_data[np.random.permutation(in_data.shape[0]), :]
        for bCounter in range(len(batch_start_ind)):
            step_time = time.time()
            bSI, bEI = batch_start_ind[bCounter], batch_end_ind[bCounter]
            dLoss, gLoss, dTime, gTime = \
                train_gan_on_batch(gModel=gModel,
                                   dModel=dModel,
                                   ganModel=ganModel,
                                   inReal=in_data[bSI:bEI, :])
            lossHistory['e'].append(eCounter)
            lossHistory['b'].append(bCounter)
            lossHistory['G'].append(gLoss)
            lossHistory['D'].append(dLoss)
            st = time.time()
            imgTime = time.time() - st
            print '... [{cS:0{fZ}d}/{tS}]) '\
                'GLoss: {gL:018.15f} ({gT:018.15f}), '\
                'DLoss: {dL:018.15f} ({dT:018.15f}), Saving {imgT:018.15f}, '\
                'Step: {stepT:018.15f}'\
                .format(fZ=len(str(len(batch_start_ind))),
                        cS=bCounter, tS=len(batch_start_ind),
                        gL=gLoss, dL=dLoss, gT=gTime, dT=dTime, imgT=imgTime,
                        stepT=time.time()-step_time)
            generate_and_save(gModel, dModel, ganModel, 1024,
                              '{:05d}-{:05d}'.format(eCounter, bCounter))
        print 'E {cE:0{fZ}}/{tE}) '\
            'GLoss: {gL:018.15f}, DLoss: {dL:018.15f}'\
            .format(fZ=len(str(num_epochs)),
                    cE=eCounter, tE=num_epochs,
                    gL=gLoss, dL=dLoss)


if __name__ == '__main__':
    np.random.seed(123)
    (trainX, trainY), (testX, testY) = load_data()
    GEN = build_generator()
    DSC = build_discriminator()
    GAN = build_gan(GEN, DSC)
    fit_gan(GEN, DSC, GAN, trainX, BATCH_SIZE, NUM_EPOCHS, shuffle=True)
    #print train_disc_on_batch(GEN, DSC, GAN, trainX[:BATCH_SIZE, :])
    #print train_gen_on_batch(GEN, DSC, GAN, BATCH_SIZE)
