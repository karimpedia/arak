#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:31:30 2017

@author: karim
"""

from __future__ import print_function

import os
import os.path as osp
import sys

from datetime import datetime

import arak
import click
import numpy as np
import pandas as pd

from arak.util.path import makedirpath, splitroot
from arak.util.timestamp import TS
from arak.util.keras_util import freeze_model, unfreeze_model
from keras.layers import Activation, BatchNormalization, Convolution1D, Input
from keras.layers import MaxPooling1D, UpSampling1D, merge, Dense, Dropout
from keras.layers import LeakyReLU, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam
from ukdale.ukdale import segmenters, ukdale

from pandas import offsets

# =============================================================================
_timestamp = datetime.now().strftime('%y%m%d%H%M%S%f')
# =============================================================================


class GANContest(object):
    def __init__(self, nb_gen=2, nb_dis=1, nb_epoch=3000, nb_gen_epoch=1,
                 nb_dis_epoch=1, batch_size=16, shuffle=True, **kwargs):
        self.nb_gen = nb_gen
        self.nb_dis = nb_dis
        self.nb_epoch = nb_epoch
        self.nb_gen_epoch = nb_gen_epoch
        self.nb_dis_epoch = nb_dis_epoch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._compiled = None
        self._dataset_initialized = False
        super(GANContest, self).__init__(**kwargs)


class MultiPlayerSingleRefree1DSum(GANContest):
    def __init__(self, **kwargs):
        self.inputX = None  # A 1d-numpy array of shape (L,)
        self.targetX = None  # A Md-numpy array of shape (L, M) where M is the
        # number of generators & L is the length of inputX
        super(self.__class__, self).__init__(nb_gen=2, nb_dis=1, **kwargs)

    def compile(self):
        if not self._dataset_initialized:
            raise "Please set the disaggregation dataset first !!!"
        self.build_generator_list()
        self.build_discriminator_list()
        self.build_adversarial_list()
        self._compiled = True
        return self

    def set_dataset(self, inputX, targetX):
        assert len(targetX) == 2  # TODO: Only two generators are supported
        assert len(inputX.shape) == 2  # (Num. segments, Segment length)
        assert all([np.array_equal(inputX.shape, x.shape) for x in targetX])
        self.inputX = inputX
        self.targetX = targetX
        self.inputX = np.expand_dims(self.inputX, -1)
        self.targetX = [np.expand_dims(x, -1) for x in self.targetX]
        self._input_shape = self.inputX.shape[1:]
        self._dataset_initialized = True
        return self

    def build_generator_list(self):
        self.Gs = [self.build_a_generator() for i in range(len(self.targetX))]
        # ---------------------------------------------------------------------
        iL = [Input(shape=(self._input_shape)) for i in range(len(self.targetX))]
        oL = merge([self.Gs[i](iL[i]) for i in range(self.nb_gen)], mode='sum', concat_axis=-1)
        # ---------------------------------------------------------------------
        model = Model(input=iL, output=oL)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3))
        self.Gs.append(model)
        # ---------------------------------------------------------------------
        for i in range(self.nb_gen)[:-1]:
            print('\n\n\n\nGenerator {}'.format(i+1))
            self.Gs[i].summary()
        print('\n\n\n\nGenerator (SUM)')
        self.Gs[-1].summary()
        return self

    def build_discriminator_list(self):
        assert self.nb_dis == 1  # TODO: Only a single refree is supported
        self.Ds = [self.build_a_discriminator() for i in range(self.nb_dis)]
        for i in range(self.nb_dis):
            print('\n\n\n\nDiscriminator {}'.format(i+1))
            self.Ds[i].summary()
        return self

    def build_adversarial_list(self):
        assert self.nb_dis == 1  # TODO: Only a single refree is supported
        assert self.nb_gen == 2  # TODO: Only two players are supported
        self.As = []
        for i in range(len(self.Ds)):
            self.As.append([])
            freeze_model(self.Ds[i])
            for j in range(len(self.Gs))[:-1]:
                iL = Input(shape=(self._input_shape))
                oL = self.Ds[i](self.Gs[j](iL))
                model = Model(input=iL, output=oL)
                print('\n\n\n\nAdversarial {}.{} [D.G]'.format(i,j))
                model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3))
                model.summary()
                self.As[-1].append(model)
            iL1 = Input(shape=(self._input_shape), name='G1')
            iL2 = Input(shape=(self._input_shape), name='G2')
            oL = self.Ds[i](self.Gs[-1]([iL1, iL2]))
            model = Model(input=[iL1, iL2], output=oL)
            print('\n\n\n\nAdversarial SUM')
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3))
            model.summary()
            self.As[-1].append(model)
        return self

    def build_a_generator(self):
        numFilters = [[32, 32], [64, 64], [128, 128], [256, 256]]
        filterLength = [[3, 3], [3, 3], [3, 3], [3, 3]]
        poolingLength = [2, 2, 3]  # Must be shorter by one
        outNumFilters = 1
        outFilterLength = 3
        dimOrdering = 'tf'
        concatAxis = 2 if dimOrdering == 'tf' else 1
        ACT = lambda x: LeakyReLU(alpha=0.3)(BatchNormalization(mode=2)(x))
        outACT = lambda x: Activation('sigmoid')(x)
        netLoss = 'binary_crossentropy'
        netOptimizer = Adam(lr=0.001)
        # ---------------------------------------------------------------------
        iL = Input(shape=(self._input_shape))
        xL = iL
        encodingLayers = []
        # ---------------------------------------------------------------------
        for i in range(len(numFilters))[:-1]:
            for j in range(len(numFilters[i])):
                xL = ACT(Convolution1D(nb_filter=numFilters[i][j],
                         filter_length=filterLength[i][j], border_mode='same')(xL))
            encodingLayers.append(xL)
            xL = MaxPooling1D(pool_length=poolingLength[i], stride=None, border_mode='valid')(xL)
        # ---------------------------------------------------------------------
        for j in range(len(numFilters[-1])):
            xL = ACT(Convolution1D(nb_filter=numFilters[-1][j],
                     filter_length=filterLength[-1][j], border_mode='same')(xL))
        # ---------------------------------------------------------------------
        for i in range(len(numFilters))[-2::-1]:
            xL = merge([UpSampling1D(length=poolingLength[i])(xL),
                        encodingLayers.pop()],
                       mode='concat', concat_axis=concatAxis)
            for j in range(len(numFilters[i]))[::-1]:
                xL = ACT(Convolution1D(nb_filter=numFilters[i][j],
                         filter_length=filterLength[i][j], border_mode='same')(xL))
        # ---------------------------------------------------------------------
        oL = outACT(Convolution1D(nb_filter=outNumFilters,border_mode='same',
                                  filter_length=outFilterLength)(xL))
        # ---------------------------------------------------------------------
        model = Model(input=iL, output=oL)
        model.compile(loss=netLoss, optimizer=netOptimizer)
        # ---------------------------------------------------------------------
        assert not encodingLayers
        return model

    def build_a_discriminator(self):
        numFilters = [[32, 32], [64, 64], [128, 128], [128, 128]]
        filterLength = [[3, 3], [3, 3], [3, 3], [3, 3]]
        poolingLength = [2, 2, 3, 3]
        denseNeurons = [512, 128]
        denseDropout = [0.3, 0.3]
        outNeurons = 4
        convACT = lambda x: LeakyReLU(alpha=0.3)(BatchNormalization(mode=2)(x))
        denseACT = lambda x: LeakyReLU(alpha=0.3)(x)
        outACT = lambda x: Activation('sigmoid')(x)
        netLoss = 'binary_crossentropy'
        netOptimizer = Adam(lr=0.001)
        # ---------------------------------------------------------------------
        iL = Input(shape=(self._input_shape))
        xL = iL
        # ---------------------------------------------------------------------
        for i in range(len(numFilters)):
            for j in range(len(numFilters[i])):
                xL = convACT(Convolution1D(nb_filter=numFilters[i][j],
                                           filter_length=filterLength[i][j],
                                           border_mode='same')(xL))
            xL = MaxPooling1D(pool_length=poolingLength[i],
                              stride=None, border_mode='valid')(xL)
        # ---------------------------------------------------------------------
        xL = Flatten()(xL)
        for i in range(len(denseNeurons)):
            xL = Dense(denseNeurons[i])(xL)
            xL = denseACT(xL)
            xL = Dropout(denseDropout[i])(xL)
        # ---------------------------------------------------------------------
        oL = outACT(Dense(outNeurons)(xL))
        # ---------------------------------------------------------------------
        model = Model(input=iL, output=oL)
        model.compile(loss=netLoss, optimizer=netOptimizer)
        return model

    def generate_all(self, fInputX, fNbClasses=None):
        fNbClasses = len(self.Gs) + 1 if fNbClasses is None else fNbClasses
        outShape = (fInputX.shape[0], fNbClasses)
        genX = [fInputX for x in self.Gs[:-1]] + [[fInputX, fInputX]]
        genY = [self.Gs[i].predict(genX[i]) for i in range(len(self.Gs))]
        disY = [np.zeros(shape=outShape, dtype=np.float32) for _ in self.Gs]
        advY = [np.ones(shape=outShape, dtype=np.float32) for _ in self.Gs]
        for i in range(len(self.Gs)):
            disY[i][:,(i+1)] = 1.0
            advY[i][:,[0, (i+1), -1]] = 0.0  # TODO
        advY[-1][:, 1:] = 0.0
        advY[-1][:, 0] = 1.0
        return genX, genY, disY, advY
        
    def _fit_discriminators(self, realX, fakeX, fakeY):
        realY = np.zeros(shape=(realX.shape[0], fakeY[0].shape[1]), dtype=np.float32)
        realY[:, 0] = 1.0
        fakeX = np.concatenate(fakeX, axis=0)
        fakeY = np.concatenate(fakeY, axis=0)
        return self._fit_a_discriminator(fModel=self.Ds[0],
                                         realX=realX, realY=realY,
                                         fakeX=fakeX, fakeY=fakeY)

    def _fit_a_discriminator(self, fModel, realX, realY, fakeX, fakeY):
        unfreeze_model(fModel)
        dataX = np.concatenate((realX, fakeX), axis=0)
        dataY = np.concatenate((realY, fakeY), axis=0)
        shfInd = np.random.permutation(dataX.shape[0])
        dataX, dataY = dataX[shfInd], dataY[shfInd]
        freeze_model(fModel)
        assert np.array_equal(dataY.shape, np.array([dataX.shape[0], len(self.Gs)+1]))
        assert np.sum(dataY) == dataX.shape[0]
        assert np.all(np.sum(dataY, axis=0) == realX.shape[0])
        assert np.all(np.sum(dataY, axis=1) == 1)
        return [fModel.train_on_batch(dataX, dataY)
                for _ in range(self.nb_dis_epoch)]

    def _fit_generators(self, dataX, dataY):
        return [self._fit_a_generator(fA=self.As[0][i], fD=self.Ds[0],
                                      dataX=dataX[i], dataY=dataY[i])
                for i in range(len(self.Gs))]

    def _fit_a_generator(self, fA, fD, dataX, dataY):
        assert np.array_equal(dataY.shape, np.array([dataY.shape[0], len(self.Gs) + 1]))
        assert np.sum(dataY) == dataY.shape[0]
        assert np.all(np.sum(dataY, axis=1) == 1)
        freeze_model(fD)
        return [fA.train_on_batch(dataX, dataY) for _ in range(self.nb_gen_epoch)]

    def fit(self):
        if not self._compiled:
            self.compile()
        fBIndS = np.arange(0, self.inputX.shape[0], self.batch_size)
        fBIndE = np.clip(fBIndS + self.batch_size, None, self.inputX.shape[0])
        fNBtch = len(fBIndS)
        for eC in range(self.nb_epoch):
            if self.shuffle:
                fInputX = self.inputX[np.random.permutation(self.inputX.shape[0])]
            for bC in range(fNBtch):
                iS, iE = fBIndS[bC], fBIndE[bC]
                print(TS('... {bC:04d}/{tC:04d} [{iS:05d}:{iE:05d}]'\
                         .format(bC=bC, tC=fNBtch, iS=iS, iE=iE)))
                fRealX = fInputX[iS:iE]
                _, fGenY, fDisY, _ = self.generate_all(fRealX)
                print(self._fit_discriminators(fRealX, fGenY, fDisY))
                fGenX, _, _, fAdvY = self.generate_all(fRealX)
                print(self._fit_generators(fGenX, fAdvY))
            self.save_models()

    def save_models(self, epoch):
        for x in range(len(self.Gs)):
            self.Gs[x].save(osp.join(self.get_out_dir('generators'),
                                     'G{}_E{:04d}'.format(x, epoch)))
        for x in range(len(self.Ds)):
            self.Ds[x].save(osp.join(self.get_out_dir('discriminators'),
                                     'D{}_E{:04d}'.format(x, epoch)))
        
    def get_out_dir(self, sub_dir=None):
        fOutD = os.getcwd()
        fOutD = osp.join(fOutD, 'tmp')
        fOutD = osp.join(fOutD, osp.split(splitroot(__file__)[-1])[0])
        fOutD = osp.join(fOutD, osp.splitext(osp.basename(__file__))[0])
        fOutD = osp.join(fOutD, _timestamp)
        fOutD = osp.join(fOutD, self.__class__.__name__)
        fOutD = osp.join(fOutD, sub_dir) if sub_dir else fOutD
        return makedirpath(fOutD)


def get_ukdale_appliance(house_id, appliance_id, start_ts, end_ts,
                         sampling_rate=6):
    fSignal = ukdale.load_appliance_hdf(house_id=house_id,
                                        appliance_id=appliance_id,
                                        start_timestamp=start_ts,
                                        end_timestamp=end_ts)
    fSignal = ukdale.resample_signal_NSecond(input_signal=fSignal,
                                             start_timestamp=start_ts,
                                             end_timestamp=end_ts,
                                             N=sampling_rate,
                                             limit=300)
    return fSignal


@click.group()
def demo():
    pass

@demo.command()
def demo_0():
    click.echo("Nothing to report")

@demo.command()
@click.option('--vis', is_flag=True)
def demo_1(vis=False):
    click.echo("Running demo-1")
    # -------------------------------------------------------------------------
    house_list = ['house_1', 'house_1']
    appliance_list = ['washing_machine', 'kettle']
    start_ts = [pd.Timestamp(2014, 1, 1, 0, 0, 0)]
    end_ts = [pd.Timestamp(2014, 6, 1, 0, 0, 0) - offsets.Second()]
    sampling_rate = 6
    signal_scaling_factor = 1e4
    segment_len = int(60 * 60 * 24 / 6)
    sliding_len = 'non-overlapping'
    # -------------------------------------------------------------------------
    fAppSig = []
    for h, p in zip(house_list, appliance_list):
        fSig = pd.DataFrame()
        for s, e in zip(start_ts, end_ts):
            fSig = pd.concat((fSig, get_ukdale_appliance(house_id=h, appliance_id=p,
                                                         start_ts=s, end_ts=e,
                                                         sampling_rate=sampling_rate)))
        fSig.sort_index(ascending=True, inplace=True, kind='quicksort')
        fAppSig.append(fSig.P.values / signal_scaling_factor)
    fAggSig = np.sum(np.array(fAppSig), axis=0)
    # -------------------------------------------------------------------------
    if vis:
        from matplotlib import pyplot as plt
        ax = plt.subplot(3, 1, 1)
        ax.plot(fAggSig)
        ax = plt.subplot(3, 1, 2, sharex=ax)
        ax.plot(fAppSig[0])
        ax = plt.subplot(3, 1, 3, sharex=ax)
        ax.plot(fAppSig[1])
        plt.show()
    # -------------------------------------------------------------------------
    fAppSig, _, _ = segmenters.segment(fAppSig, segment_len=segment_len, sliding=sliding_len)
    fAggSig, _, _ = segmenters.segment(fAggSig, segment_len=segment_len, sliding=sliding_len)
    # -------------------------------------------------------------------------
    net = MultiPlayerSingleRefree1DSum()
    net.set_dataset(inputX=fAggSig, targetX=fAppSig)
    net.compile()
    net.fit()

    
cClickCollection = click.CommandCollection(sources=[demo])

if __name__ == '__main__':
    cClickCollection()
