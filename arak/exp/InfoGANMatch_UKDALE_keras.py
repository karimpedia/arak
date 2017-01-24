#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 04:58:34 2017

@author: karim
"""


from __future__ import print_function


import os
import os.path as osp
import sys

from collections import OrderedDict
from collections import Iterable
from datetime import datetime

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, merge
from keras.layers import ELU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, UpSampling2D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
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
    def __init__(self, num_gen=2, code_shape=(10*60*24,),
                 batch_size=32, num_epochs=3, gen_epochs=1,
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
        self.gen_models = None
        self.dis_models = None
        self.adv_models = None
        self.trainX, trainY = None
        self.testX, testY = None
        self.appliance_signals = OrderedDict()
        self.compiled = False

    def compile(self):
        self._init_data()
        self._build_generator_list()
        self._build_discriminator_list()
        self._build_adversarial_list()
        self.compiled = True
        return self

    def _init_data(self):
        import ukdale
        import pandas as pd
        from pandas import offsets
        house_id = 'house_1'
        appliance_a = 'washing_machine'
        appliance_b = 'dishwasher'
        sg_samplingInterval = 6  # Seconds
        cnn_normalization = 1e4
        start_ts = \
            [pd.Timestamp(2014, 1, 1, 0, 0, 0),
             pd.Timestamp(2016, 1, 1, 0, 0, 0)]
        end_ts = \
            [pd.Timestamp(2014, 2, 1, 0, 0, 0) - offsets.Second(),
             pd.Timestamp(2016, 2, 1, 0, 0, 0) - offsets.Second()]
        # ---------------------------------------------------------------------
        for x in self.appliance_list:
            self.appliance_signals = OrderedDict()
            self.appliance_signals[x] = \
                [_load_ukdale_appliance(self.house_id, x, y, z)
                 for y, z in zip(self.start_ts, self.end_ts)]
            self.app
            
        
        
        
        
        
        cAAppTrain = [self._load_ukdale_appliance
                      ukdale.load_appliance_hdf(house_id, appliance_a,
                                                x, y, load_boundaries=True)
                      for x, y in zip(start_timestamps_train,
                                      end_timestamps_train)]
        cBAppTrain = [ukdale.load_appliance_hdf(house_id, appliance_b,
                                                x, y, load_boundaries=True)
                      for x, y in zip(start_timestamps_train,
                                      end_timestamps_train)]
        # ---------------------------------------------------------------------
        cAAppValid = [ukdale.load_appliance_hdf(house_id, appliance_a,
                                                x, y, load_boundaries=True)
                      for x, y in zip(start_timestamps_valid,
                                      end_timestamps_valid)]
        cBAppValid = [ukdale.load_appliance_hdf(house_id, appliance_b,
                                                x, y, load_boundaries=True)
                      for x, y in zip(start_timestamps_valid,
                                      end_timestamps_valid)]
        # ---------------------------------------------------------------------
        cAAppTrain = \
            [ukdale.resample_signal_NSecond(input_signal=x,
                                            start_timestamp=y,
                                            end_timestamp=z,
                                            N=6, limit=300)
             for x, y, z in zip(cAAppTrain,
                                start_timestamps_train,
                                end_timestamps_train)]
        cBAppTrain = \
            [ukdale.resample_signal_NSecond(input_signal=x,
                                            start_timestamp=y,
                                            end_timestamp=z,
                                            N=6, limit=300)
             for x, y, z in zip(cBAppTrain,
                                start_timestamps_train,
                                end_timestamps_train)]
        # ---------------------------------------------------------------------
        cAAppValid = \
            [ukdale.resample_signal_NSecond(input_signal=x,
                                            start_timestamp=y,
                                            end_timestamp=z,
                                            N=6, limit=300)
             for x, y, z in zip(cAAppValid,
                                start_timestamps_valid,
                                end_timestamps_valid)]
        cBAppValid = \
            [ukdale.resample_signal_NSecond(input_signal=x,
                                            start_timestamp=y,
                                            end_timestamp=z,
                                            N=6, limit=300)
             for x, y, z in zip(cBAppValid,
                                start_timestamps_valid,
                                end_timestamps_valid)]
        # ---------------------------------------------------------------------
        cAAppTrainDF = pd.DataFrame()
        for x in cAAppTrainDF:
            cAAppTrainDF = pd.concat((cAAppTrainDF, x))
        cBAppTrain = pd.DataFrame()
        for x in cBAppTrainDF:
            cBAppTrain = pd.concat((cBAppTrain, x))
        # -------------------------------------------------------------------------
        cRefAppValid = pd.DataFrame()
        for x in cRefAppValidDF:
            cRefAppValid = pd.concat((cRefAppValid, x))
        cRawAppValid = pd.DataFrame()
        for x in cRawAppValidDF:
            cRawAppValid = pd.concat((cRawAppValid, x))
        # -------------------------------------------------------------------------
        cRefAppTrain.sort_index(ascending=True, inplace=True, kind='quicksort')
        cRawAppTrain.sort_index(ascending=True, inplace=True, kind='quicksort')
        cRefAppValid.sort_index(ascending=True, inplace=True, kind='quicksort')
        cRawAppValid.sort_index(ascending=True, inplace=True, kind='quicksort')
        # -------------------------------------------------------------------------
        print _cDispSectionBreaker
        return cRefAppTrain.P.values, cRawAppTrain.P.values, \
        cRefAppValid.P.values, cRawAppValid.P.values


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

    def _load_ukdale_appliance(house_id, appliane_id, tsStart, tsEnd):
        x = ukdale.load_appliance_hdf(house_id, appliance_id, tsStart, tsEnd,
                                      load_boundaries=True)
        return ukdale.resample_signal_NSecond(input_signal=x,
                                              start_timestamp=tsStart,
                                              end_timestamp=tsEnd,
                                              N=6, limit=300)

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

        c01 = Convolution1D(nb_filter=16, filter_length=3, border_mode='same', dim_ordering='tf')(inLayer)
        c01 = Activation('relu')(BatchNormalization(mode=2)(c01))
        c01 = Convolution1D(nb_filter=16, filter_length=3, border_mode='same', dim_ordering='tf')(c01)
        c01 = Activation('relu')(BatchNormalization(mode=2)(c01))
        p01 = MaxPooling1D(pool_length=2, stride=None, border_mode='valid', dim_ordering='tf')(c01)
        
        c02 = Convolution1D(nb_filter=32, filter_length=3, border_mode='same', dim_ordering='tf')(p01)
        c02 = Activation('relu')(BatchNormalization(mode=2)(c02))
        c02 = Convolution1D(nb_filter=32, filter_length=3, border_mode='same', dim_ordering='tf')(c02)
        c02 = Activation('relu')(BatchNormalization(mode=2)(c02))
        p02 = MaxPooling1D(pool_length=2, stride=None, border_mode='valid', dim_ordering='tf')(c02)

        c03 = Convolution1D(nb_filter=64, filter_length=3, border_mode='same', dim_ordering='tf')(p02)
        c03 = Activation('relu')(BatchNormalization(mode=2)(c03))
        c03 = Convolution1D(nb_filter=64, filter_length=3, border_mode='same', dim_ordering='tf')(c03)
        c03 = Activation('relu')(BatchNormalization(mode=2)(c03))
        p03 = MaxPooling1D(pool_length=3, stride=None, border_mode='valid', dim_ordering='tf')(c03)

        c04 = Convolution1D(nb_filter=128, filter_length=3, border_mode='same', dim_ordering='tf')(p03)
        c04 = Activation('relu')(BatchNormalization(mode=2)(c04))
        c04 = Convolution1D(nb_filter=128, filter_length=3, border_mode='same', dim_ordering='tf')(c04)
        c04 = Activation('relu')(BatchNormalization(mode=2)(c04))
        
        m05 = merge([UpSampling1D(length=3)(c04), c03], mode='concat', concat_axis=3)
        c05 = Convolution1D(nb_filter=64, filter_length=3, border_mode='same', dim_ordering='tf')(m05)
        c05 = Activation('relu')(BatchNormalization(mode=2)(c05))
        c05 = Convolution1D(nb_filter=64, filter_length=3, border_mode='same', dim_ordering='tf')(c05)
        c05 = Activation('relu')(BatchNormalization(mode=2)(c05))

        m06 = merge([UpSampling1D(length=2)(c05), c02], mode='concat', concat_axis=3)
        c06 = Convolution1D(nb_filter=32, filter_length=3, border_mode='same', dim_ordering='tf')(m06)
        c06 = Activation('relu')(BatchNormalization(mode=2)(c06))
        c06 = Convolution1D(nb_filter=32, filter_length=3, border_mode='same', dim_ordering='tf')(c06)
        c06 = Activation('relu')(BatchNormalization(mode=2)(c06))

        m07 = merge([UpSampling1D(length=2)(c06), c01], mode='concat', concat_axis=3)
        c07 = Convolution1D(nb_filter=16, filter_length=3, border_mode='same', dim_ordering='tf')(m07)
        c07 = Activation('relu')(BatchNormalization(mode=2)(c07))
        c07 = Convolution1D(nb_filter=16, filter_length=3, border_mode='same', dim_ordering='tf')(c07)
        c07 = Activation('relu')(BatchNormalization(mode=2)(c07))

        outLayer = Activation('sigmoid')(Convolution1D(nb_filter=1, filter_length=3, border_mode='same', dim_ordering='tf')(c07))

        model = Model(input=inLayer, output=outLayer)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
        return model

    def _build_discriminator(self):
        inLayerA = Input(shape=self.code_shape)
        inLayerB = Input(shape=self.code_shape)

        x = merge([inLayerA, inLayerB], mode='concat', concat_axis=3)

        x = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode='same',
                          init='he_normal', dim_ordering='tf')(x)
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
#model._generate_all(100)
#model.fit()

#print [model._generate(x, 100) for x in model.gen_models]
