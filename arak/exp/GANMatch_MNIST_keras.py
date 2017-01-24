#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 21:06:51 2017

@author: karim
"""

from __future__ import print_function


import os
import os.path as osp
import sys

from collections import Iterable
from datetime import datetime

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, merge
from keras.layers import ELU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import Adam, SGD
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

import numpy as np
from PIL import Image
import argparse
import math

import keras
import keras.backend as K

from arak.util.path import makedirpath, splitroot


# ================================================================= Identifiers
_timestamp = datetime.now().strftime('%y%m%d%H%M%S%f')
_outDirPath = osp.join(os.getcwd(), 'tmp',
                       osp.split(splitroot(__file__)[-1])[0],
                       osp.splitext(osp.basename(__file__))[0], _timestamp)
# =============================================================================


def set_trainable(model, trainable):
    model.trainable = trainable
    for x in model.layers:
        x.trainable = trainable


def freeze_model(model):
    set_trainable(model, False)


def unfreeze_model(model):
    set_trainable(model, True)




class MultiGSingleD(object):
    def __init__(self, num_gen=2, code_shape=(100,),
                 batch_size=100, num_epochs=300, gen_epochs=1,
                 dis_epochs=1, shuffle=True, seed=123, **kwargs):
        self.num_gen = num_gen
        self.num_dis = 1  # TODO: Single discriminator
        self.code_shape = \
            code_shape if isinstance(code_shape, Iterable) else (code_shape,)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gen_epochs = gen_epochs
        self.dis_epochs = dis_epochs
        self.shuffle = shuffle
        self.seed = seed
        self.IMG_SHAPE = (28, 28)
        self.gen_models = None
        self.dis_models = None
        self.adv_models = None
        self.trainX = None
        self.compiled = False

    def compile(self):
        self._init_data()
        self._build_generator_list()
        self._build_discriminator_list()
        self._build_adversarial_list()
        self.compiled = True
        return self

    def _build_generator_list(self):
        self.gen_models = \
            tuple([self._build_generator() for i in range(self.num_gen)])
        for i in range(self.num_gen):
            print('\n\n\n\nGenerator {}'.format(i))
            self.gen_models[i].summary()
        return self

    def _build_discriminator_list(self):
        assert self.num_dis == 1  # TODO: Single discriminator
        self.dis_models = \
            tuple([self._build_discriminator() for i in range(self.num_dis)])
        for i in range(self.num_dis):
            print('\n\n\n\nDiscriminator {}'.format(i))
            self.dis_models[i].summary()
        return self

    def _build_adversarial_list(self):
        assert self.num_dis == 1  # TODO: Single discriminator
        assert self.gen_models  # That they are built
        assert self.dis_models  # That they are built
        self.adv_models = \
            tuple([tuple([self._build_adversarial(fGEN, fDIS)
                          for fGEN in self.gen_models])
                   for fDIS in self.dis_models])
        for i in range(len(self.adv_models)):
            for j in range(len(self.adv_models[i])):
                print('\n\n\n\nAdversarial {}.{}'.format(i, j))
                print('........... D.G')
                self.adv_models[i][j].summary()
        return self

    def _build_generator(self):
        inLayer = Input(shape=self.code_shape)
        x = Dense(output_dim=1024, init='he_normal')(inLayer)
        x = BatchNormalization(mode=2)(x)
        x = Activation('sigmoid')(x)
        x = Dense(7 * 7 * 64, init='he_normal')(x)
        x = BatchNormalization(mode=2)(x)
        x = Activation('sigmoid')(x)
        x = Reshape((7, 7, 64))(x)
        x = UpSampling2D(size=(2, 2), dim_ordering='tf')(x)
        x = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, border_mode='same',
                          init='he_normal', dim_ordering='tf')(x)
        x = Activation('sigmoid')(x)
        x = UpSampling2D(size=(2, 2), dim_ordering='tf')(x)
        x = Convolution2D(nb_filter=1, nb_row=3, nb_col=3, border_mode='same',
                          init='he_normal', dim_ordering='tf')(x)
        outLayer = Activation('sigmoid')(x)
        model = Model(input=inLayer, output=outLayer)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
        return model

    def _build_discriminator(self):
        inLayer = Input(shape=(self.IMG_SHAPE + (1,)))
        x = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='same',
                          init='he_normal', dim_ordering='tf')(inLayer)
        x = Activation('sigmoid')(x)
        x = MaxPooling2D(pool_size=(2, 2), border_mode='valid',
                         dim_ordering='tf')(x)
        x = Convolution2D(nb_filter=32, nb_row=3, nb_col=3, border_mode='same',
                          init='he_normal', dim_ordering='tf')(x)
        x = Activation('sigmoid')(x)
        x = MaxPooling2D(pool_size=(2, 2), border_mode='valid',
                         dim_ordering='tf')(x)
        x = Flatten()(x)
        x = Dense(1024, init='he_normal')(x)
        x = Activation('sigmoid')(x)
        x = Dense(3, init='he_normal')(x)
        outLayer = Activation('sigmoid')(x)
        model = Model(input=inLayer, output=outLayer)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
        return model

    def _build_adversarial(self, fGEN, fDIS):
        fDIS.trainable = False
        inLayer = Input(shape=self.code_shape)
        outLayer = fDIS(fGEN(inLayer))
        model = Model(input=inLayer, output=outLayer)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
        return model

    def _init_data(self):
        (trainX, trainY), (testX, testY) = mnist.load_data()
        trainX = trainX.astype(np.float32) / 255.0
        testX = testX.astype(np.float32) / 255.0
        trainX = trainX.reshape((trainX.shape[0],) + self.IMG_SHAPE + (1,))
        testX = testX.reshape((testX.shape[0],) + self.IMG_SHAPE + (1,))
        self.trainX = trainX
        self.testX = testX
        self.trainY = trainY
        self.testY = testY
        return self

    def _generate_all(self, num_samples):
        genX = np.random.uniform(0., 1., ((num_samples,) + self.code_shape))
#        genX = tuple([np.random.uniform(0.,
#                                        1.,
#                                        ((num_samples,) + self.code_shape))
#                      for x in range(self.num_gen)])
        genX = tuple([genX for x in range(self.num_gen)])
        genY = tuple([self.gen_models[i].predict(genX[i], verbose=False)
                      for i in range(self.num_gen)])
        disY = tuple([float(x+1) * np.ones(num_samples, dtype=np.float32)
                      for x in range(self.num_gen)])
        advY = tuple([np.zeros(num_samples, dtype=np.float32)
                      for x in range(self.num_gen)])
        return genX, genY, disY, advY

    def _generate(self, fGEN, num_samples):
        genX = np.random.uniform(0., 1., ((num_samples,) + self.code_shape))
        genY = fGEN.predict(genX, verbose=False)
        return genX, genY

    def _fit_dis_list(self, realX, fakeX, fakeY):
        assert len(self.dis_models) == 1  # TODO: Single discriminator
        fakeX = np.concatenate(fakeX, axis=0)
        fakeY = np.concatenate(fakeY, axis=0)
        realY = np.zeros(realX.shape[0])
        return self._fit_dis_model(fDIS=self.dis_models[0],
                                   realX=realX,
                                   realY=realY,
                                   fakeX=fakeX,
                                   fakeY=fakeY)  # TODO: Single discriminator

    def _fit_dis_model(self, fDIS, realX, realY, fakeX, fakeY):
        unfreeze_model(fDIS)
        dataX = np.concatenate((realX, fakeX), axis=0)
        dataY = np.concatenate((realY, fakeY), axis=0)
        dataY = to_categorical(dataY, fDIS.layers[-1].output_shape[1])
        assert np.array_equal(dataY.shape,
                              np.array([dataX.shape[0], self.num_gen+1]))
        assert np.sum(dataY) == dataX.shape[0]
        assert np.all(np.sum(dataY, axis=0) == realX.shape[0])
        assert np.all(np.sum(dataY, axis=1) == 1)
        loss_list = []
        for x in range(self.dis_epochs):
            cLoss = fDIS.train_on_batch(dataX, dataY)
            loss_list.append(cLoss)
        return loss_list

    def _fit_gen_list(self, dataX, dataY):
        assert len(self.adv_models) == 1  # TODO: Single discriminator
        assert len(self.adv_models[0]) == self.num_gen
        loss_list = []
        for i in range(self.num_gen):
            # TODO: Single discriminator
            cLoss = self._fit_gen_model(fGEN=self.gen_models[i],
                                        fDIS=self.dis_models[0],
                                        fADV=self.adv_models[0][i],
                                        dataX=dataX[i],
                                        dataY=dataY[i])
            loss_list.append(cLoss)
        return loss_list

    def _fit_gen_model(self, fGEN, fDIS, fADV, dataX, dataY):
        freeze_model(fDIS)
        dataY = to_categorical(dataY, fADV.layers[-1].output_shape[1])
        assert np.array_equal(dataY.shape,
                              np.array([dataY.shape[0], self.num_gen+1]))
        assert np.sum(dataY) == dataY.shape[0]
        assert np.sum(dataY) == np.sum(dataY, axis=0)[0]
        assert np.all(np.sum(dataY, axis=1) == 1)
        loss_list = []
        for x in range(self.gen_epochs):
            cLoss = fADV.train_on_batch(dataX, dataY)
            loss_list.append(cLoss)
        return loss_list

    def fit(self):
        if not self.compiled:
            self._compile()
        fBIndS = np.arange(0, self.trainX.shape[0], self.batch_size)
        fBIndE = np.clip(fBIndS + self.batch_size, None, self.trainX.shape[0])
        fNumBatches = len(fBIndS)
        for eC in range(self.num_epochs):
            print('Epoch {c:0{z}d}/{t:d} ...'
                  .format(c=eC,
                          t=self.num_epochs,
                          z=len(str(self.num_epochs))))
            epochX = self.trainX
            if self.shuffle:
                epochX = \
                    self.trainX[np.random.permutation(self.trainX.shape[0])]
            for bC in range(fNumBatches):
                iS, iE = fBIndS[bC], fBIndE[bC]
                outString = '... {c:0{z}d}/{t:d}) [{iS:05d}:{iE:05d}]'\
                    .format(c=bC, t=fNumBatches, z=len(str(fNumBatches)),
                            iS=iS, iE=iE)
                realX = epochX[iS:iE]
                print(outString)
                genX, genY, disY, advY = \
                    self._generate_all(realX.shape[0])
                self._fit_dis_list(realX, genY, disY)
                genX, genY, disY, advY = \
                    self._generate_all(realX.shape[0]*3)
                self._fit_gen_list(genX, advY)
                print('-'*150)
                if bC % 10 == 0:
                    self.save_generated_samples(100, eC, bC)
            self.save_models(eC)
            print('='*200)

    def save_generated_samples(self, num_samples, eC, bC):
        _, genY, _, _ = self._generate_all(num_samples)
        imgList = [self.combine_images(np.squeeze(x)) for x in genY]
        for i in range(self.num_gen):
            img = imgList[i]
            fileName = '{:03d}_{:05d}.png'.format(eC, bC)
            filePath = osp.join(self.get_out_dir(str(i)), fileName)
            Image.fromarray(img.astype(np.uint8)).save(filePath)

    def combine_images(self, generated_images):
        width = int(math.sqrt(generated_images.shape[0]))
        height = int(math.ceil(float(generated_images.shape[0])/width))
        image = np.zeros((height*self.IMG_SHAPE[0],
                          width*self.IMG_SHAPE[1]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*self.IMG_SHAPE[0]:(i+1)*self.IMG_SHAPE[0],
                  j*self.IMG_SHAPE[1]:(j+1)*self.IMG_SHAPE[1]] = img * 255.0
        return image

    def save_models(self, eC):
        for i in range(self.num_dis):
            fileName = 'DIS{:02d}_E{:05d}.hdf'.format(i, eC)
            self.dis_models[i].save(osp.join(self.get_out_dir('models'),
                                             fileName))
        for i in range(self.num_gen):
            fileName = 'GEN{:02d}_E{:05d}.hdf'.format(i, eC)
            self.gen_models[i].save(osp.join(self.get_out_dir('models'),
                                             fileName))
        assert self.num_dis == 1
        testEst = self.dis_models[0].predict(self.testX)[:, 1:]
        testEst = np.argmax(testEst, axis=1)
        testEst = np.concatenate((testEst[:, np.newaxis],
                                  self.testY[:, np.newaxis]), axis=1)
        fileName = 'EST{:02d}_E{:05d}.npy'.format(0, eC)
        np.save(osp.join(self.get_out_dir('eval'), fileName), testEst)

    def get_out_dir(self, sub_dir):
        if sub_dir:
            base_dir = osp.join(_outDirPath, self.__class__.__name__)
            makedirpath(osp.join(base_dir, sub_dir))
            return osp.join(base_dir, sub_dir)
        else:
            base_dir = osp.join(_outDirPath, self.__class__.__name__)
            makedirpath(base_dir)
            return base_dir


model = MultiGSingleD(num_epochs=300, dis_epochs=2, batch_size=100)
model.compile()
model._generate_all(100)
model.fit()

#print [model._generate(x, 100) for x in model.gen_models]
