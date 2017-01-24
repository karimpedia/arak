#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 21:47:53 2017

@author: karim
"""

import os
import os.path as osp
import sys

from datetime import datetime

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, merge
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import Adam, SGD
from keras.datasets import mnist
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
makedirpath(_outDirPath)
# =============================================================================


class InfoGAN(object):
    def __init__(self, GModel, DModel, QDist, **kwargs):
        super(InfoGAN, self).__init__(**kwargs)
        self.QDist = QDist
        self.GModel = GModel
        self.DModel = DModel
        self.out_dir = osp.join(_outDirPath, 'InfoGAN')

    def fit(self, in_data, num_j_epochs=1000, num_batch=128, num_max=None,
            num_vis=200, num_k_epochs=(1, 2),
            optimizer_g=Adam(lr=1e-3), optimizer_d=Adam(lr=2e-4)):
        num_max = num_batch * 100 if num_max is None else num_max
        G, D, Q = self.GModel, self.DModel, self.QDist
        hidden_D = Sequential([l for l in D.layers[-1]])

        inpt_G = Input(D.input_shape[1:])
        inpt_R = Input(D.input_shape[1:])
        code_G = Input(G.input_shape[1:])

        D2Batch = Model([inpt_G, inpt_R], [D(inpt_G), D(inpt_R)])
        neg_log_Q = -G.g_info.register(G.get_info_coding(code_G),
                                       Q(hidden_D(G(code_G))))

        mut_info = K.function([code_G], [neg_log_Q])
        res_info = mut_info(G.sample())
        print res_info[0].shape
        
        



























































class TwoClassBinaryBasicModelMututal():
    def __init__(self, batch_size, num_epochs, **kwargs):
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def compille(self):
        self._init_mnist()
        self._init_generatorA()
        self._init_generatorB()
        self._init_discriminator()
        self._init_gan()
        self.compiled = True
        self.AG.compile(loss='binary_crossentropy', optimizer="SGD")
        self.BG.compile(loss='binary_crossentropy', optimizer="SGD")
        self.GAN.compile(loss='binary_crossentropy',
                         optimizer=SGD(lr=5e-4, momentum=0.9, nesterov=True))
        self.D.trainable = True
        self.D.compile(loss='binary_crossentropy',
                       optimizer=SGD(lr=5e-4, momentum=0.9, nesterov=True))
        return self

    def _init_mnist(self):
        (trainX, _), _ = mnist.load_data()
        trainX = (trainX.astype(np.float32) - 127.5) / 127.5
        trainX = trainX.reshape((trainX.shape[0], 1) + trainX.shape[1:])
        self.trainX = trainX
        self.num_batches = int(self.trainX.shape[0] / self.batch_size)
        return self

    def _init_generatorA(self):
        model = Sequential()
        model.add(Dense(input_dim=100, output_dim=1024))
        model.add(Activation('tanh'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(64, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(1, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        self.AG = model
        return self
        
    def _init_generatorB(self):
        model = Sequential()
        model.add(Dense(input_dim=100, output_dim=1024))
        model.add(Activation('tanh'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(64, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(1, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        self.BG = model
        return self
        

    def _init_discriminator(self):
        model = Sequential()
        model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=(1, 28, 28)))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, 5, 5))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('tanh'))
        model.add(Dense(3))
        model.add(Activation('sigmoid'))
        self.D = model
        return self

    def _init_gan(self):
        i1 = Input(shape=(100,))
        i2 = Input(shape=(100,))
        g1 = self.AG(i1)
        g2 = self.BG(i2)
        m1 = merge([g1, g2], mode='sum')
        self.D.trainable = False
        O = self.D(m1)
        self.GAN = Model(input=[i1, i2], output=O)
        return self


    def train(self):
        if not self.compiled:
            self.compille()
        noise = np.zeros((self.batch_size, 100))
        for epoch in range(self.num_epochs):
            print 'Epoch {c:0{z}d}/{t:d}'.format(c=epoch, t=self.num_epochs, z=len(str(self.num_epochs)))
            for batch_ind in range(self.num_batches):
                outString = '... {c:0{z}d}/{t:d}) '.format(c=batch_ind, t=self.num_batches, z=len(str(self.num_batches)))

                self.D.trainable = True

                for sample_ind in range(self.batch_size):
                    noise[sample_ind, :] = np.random.uniform(-1, 1, 100)
                realX = self.trainX[batch_ind * self.batch_size: (batch_ind+1)*self.batch_size]
                fakeXA = self.AG.predict(noise, verbose=0)
                fakeXB = self.BG.predict(noise, verbose=0)
                discX = np.concatenate((realX, fakeXA, fakeXB))
                discY = np.array([0] * self.batch_size + [1] * self.batch_size + [2] * self.batch_size)
                discY = keras.utils.np_utils.to_categorical(discY, 3)
                discL = self.D.train_on_batch(discX, discY)

                self.D.trainable = False

                for i in range(self.batch_size):
                    noise[i, :] = np.random.uniform(-1, 1, 100)
                ganY = np.array([0] * self.batch_size)
                ganY = keras.utils.np_utils.to_categorical(ganY, 3)
                ganL = self.GAN.train_on_batch([noise, noise], ganY)

                outString += 'D: {:0.10f}, '.format(discL)
                outString += 'G: {:0.10f}, '.format(ganL)
                print outString
                if batch_ind % 10 == 0:
                    image = self.combine_images(fakeXA)
                    image = image * 127.5 + 127.5
                    Image.fromarray(image.astype(np.uint8)).save(
                        osp.join(_outDirPath, 'A.{:07d}.{:07d}.png'
                                              .format(epoch, batch_ind)))
                    image = self.combine_images(fakeXB)
                    image = image * 127.5 + 127.5
                    Image.fromarray(image.astype(np.uint8)).save(
                        osp.join(_outDirPath, 'B.{:07d}.{:07d}.png'
                                              .format(epoch, batch_ind)))
#                if index % 10 == 9:
#                    generator.save_weights('generator', True)
#                    discriminator.save_weights('discriminator', True)
#                sys.exit(0)

    def combine_images(self, generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[2:]
        image = np.zeros((height*shape[0], width*shape[1]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0],
                  j*shape[1]:(j+1)*shape[1]] = img[0, :, :]
        return image



TwoClassBinaryBasicModelMututal(256, 300).compille().train()
sys.exit(0)



class TwoClassBinaryBasicModel():
    def __init__(self, batch_size, num_epochs, **kwargs):
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def compille(self):
        self._init_mnist()
        self._init_generatorA()
        self._init_generatorB()
        self._init_discriminator()
        self._init_ganA()
        self._init_ganB()
        self.compiled = True
        self.AG.compile(loss='binary_crossentropy', optimizer="SGD")
        self.AGAN.compile(loss='binary_crossentropy',
                          optimizer=SGD(lr=5e-4, momentum=0.9, nesterov=True))
        self.BG.compile(loss='binary_crossentropy', optimizer="SGD")
        self.BGAN.compile(loss='binary_crossentropy',
                          optimizer=SGD(lr=5e-4, momentum=0.9, nesterov=True))
        self.D.trainable = True
        self.D.compile(loss='binary_crossentropy',
                       optimizer=SGD(lr=5e-4, momentum=0.9, nesterov=True))
        return self

    def _init_mnist(self):
        (trainX, _), _ = mnist.load_data()
        trainX = (trainX.astype(np.float32) - 127.5) / 127.5
        trainX = trainX.reshape((trainX.shape[0], 1) + trainX.shape[1:])
        self.trainX = trainX
        self.num_batches = int(self.trainX.shape[0] / self.batch_size)
        return self

    def _init_generatorA(self):
        model = Sequential()
        model.add(Dense(input_dim=100, output_dim=1024))
        model.add(Activation('tanh'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(64, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(1, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        self.AG = model
        return self
        
    def _init_generatorB(self):
        model = Sequential()
        model.add(Dense(input_dim=100, output_dim=1024))
        model.add(Activation('tanh'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(64, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(1, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        self.BG = model
        return self
        

    def _init_discriminator(self):
        model = Sequential()
        model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=(1, 28, 28)))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, 5, 5))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('tanh'))
        model.add(Dense(3))
        model.add(Activation('sigmoid'))
        self.D = model
        return self

    def _init_ganA(self):
        model = Sequential()
        model.add(self.AG)
        self.D.trainable = False
        model.add(self.D)
        self.AGAN = model
        return self

    def _init_ganB(self):
        model = Sequential()
        model.add(self.BG)
        self.D.trainable = False
        model.add(self.D)
        self.BGAN = model
        return self

    def train(self):
        if not self.compiled:
            self.compille()
        noise = np.zeros((self.batch_size, 100))
        for epoch in range(self.num_epochs):
            print 'Epoch {c:0{z}d}/{t:d}'.format(c=epoch, t=self.num_epochs, z=len(str(self.num_epochs)))
            for batch_ind in range(self.num_batches):
                outString = '... {c:0{z}d}/{t:d}) '.format(c=batch_ind, t=self.num_batches, z=len(str(self.num_batches)))
                self.D.trainable = True
                for sample_ind in range(self.batch_size):
                    noise[sample_ind, :] = np.random.uniform(-1, 1, 100)
                realX = self.trainX[batch_ind * self.batch_size: (batch_ind+1)*self.batch_size]
                fakeXA = self.AG.predict(noise, verbose=0)
                fakeXB = self.BG.predict(noise, verbose=0)
                discX = np.concatenate((realX, fakeXA, fakeXB))
                discY = np.array([0] * self.batch_size + [1] * self.batch_size + [2] * self.batch_size)
                discY = keras.utils.np_utils.to_categorical(discY, 3)
                discL = self.D.train_on_batch(discX, discY)
                discL = self.D.train_on_batch(discX, discY)
                self.D.trainable = False

                for i in range(self.batch_size):
                    noise[i, :] = np.random.uniform(-1, 1, 100)
                ganYA = np.array([0] * self.batch_size)
                ganYA = keras.utils.np_utils.to_categorical(ganYA, 3)
                ganLA = self.AGAN.train_on_batch(noise, ganYA)

                for i in range(self.batch_size):
                    noise[i, :] = np.random.uniform(-1, 1, 100)
                ganYB = np.array([0] * self.batch_size)
                ganYB = keras.utils.np_utils.to_categorical(ganYB, 3)
                ganLB = self.BGAN.train_on_batch(noise, ganYB)

                outString += 'D: {:0.10f}, '.format(discL)
                outString += 'Ga: {:0.10f}, '.format(ganLA)
                outString += 'Gb: {:0.10f}, '.format(ganLB)
                print outString
                if batch_ind % 10 == 0:
                    image = self.combine_images(fakeXA)
                    image = image * 127.5 + 127.5
                    Image.fromarray(image.astype(np.uint8)).save(
                        osp.join(_outDirPath, 'A.{:07d}.{:07d}.png'\
                                              .format(epoch, batch_ind)))
                    image = self.combine_images(fakeXB)
                    image = image * 127.5 + 127.5
                    Image.fromarray(image.astype(np.uint8)).save(
                        osp.join(_outDirPath, 'B.{:07d}.{:07d}.png'\
                                              .format(epoch, batch_ind)))
#                if index % 10 == 9:
#                    generator.save_weights('generator', True)
#                    discriminator.save_weights('discriminator', True)
#                sys.exit(0)

    def combine_images(self, generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[2:]
        image = np.zeros((height*shape[0], width*shape[1]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0],
                  j*shape[1]:(j+1)*shape[1]] = img[0, :, :]
        return image



TwoClassBinaryBasicModel(256, 300).compille().train()
sys.exit(0)






























































class BinaryBasicModel():
    def __init__(self, batch_size, num_epochs, **kwargs):
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def compille(self):
        self._init_mnist()
        self._init_generator()
        self._init_discriminator()
        self._init_gan()
        self.compiled = True
        self.G.compile(loss='binary_crossentropy', optimizer="SGD")
        self.GAN.compile(loss='binary_crossentropy',
                         optimizer=SGD(lr=5e-4, momentum=0.9, nesterov=True))
        self.D.trainable = True
        self.D.compile(loss='binary_crossentropy',
                       optimizer=SGD(lr=5e-4, momentum=0.9, nesterov=True))
        return self

    def _init_mnist(self):
        (trainX, _), _ = mnist.load_data()
        trainX = (trainX.astype(np.float32) - 127.5) / 127.5
        trainX = trainX.reshape((trainX.shape[0], 1) + trainX.shape[1:])
        self.trainX = trainX
        self.num_batches = int(self.trainX.shape[0] / self.batch_size)
        return self

    def _init_generator(self):
        model = Sequential()
        model.add(Dense(input_dim=100, output_dim=1024))
        model.add(Activation('tanh'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(64, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(1, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        self.G = model
        return self
        

    def _init_discriminator(self):
        model = Sequential()
        model.add(Convolution2D(64, 5, 5, border_mode='same', input_shape=(1, 28, 28)))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, 5, 5))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('tanh'))
        model.add(Dense(2))
        model.add(Activation('sigmoid'))
        self.D = model
        return self

    def _init_gan(self):
        model = Sequential()
        model.add(self.G)
        self.D.trainable = False
        model.add(self.D)
        self.GAN = model
        return self

    def train(self):
        if not self.compiled:
            self.compille()
        noise = np.zeros((self.batch_size, 100))
        for epoch in range(self.num_epochs):
            print 'Epoch {c:0{z}d}/{t:d}'.format(c=epoch, t=self.num_epochs, z=len(str(self.num_epochs)))
            for batch_ind in range(self.num_batches):
                outString = '... {c:0{z}d}/{t:d}) '.format(c=batch_ind, t=self.num_batches, z=len(str(self.num_batches)))
                for sample_ind in range(self.batch_size):
                    noise[sample_ind, :] = np.random.uniform(-1, 1, 100)
                self.D.trainable = True
                realX = self.trainX[batch_ind * self.batch_size: (batch_ind+1)*self.batch_size]
                fakeX = self.G.predict(noise, verbose=0)
                discX = np.concatenate((realX, fakeX))
                discY = np.array([1] * self.batch_size + [0] * self.batch_size)
                discY = keras.utils.np_utils.to_categorical(discY, 2)
                discL = self.D.train_on_batch(discX, discY)
                self.D.trainable = False
                for i in range(self.batch_size):
                    noise[i, :] = np.random.uniform(-1, 1, 100)
                ganY = np.array([1] * self.batch_size)
                ganY = keras.utils.np_utils.to_categorical(ganY, 2)
                ganL = self.GAN.train_on_batch(noise, ganY)
                outString += 'D: {:0.10f}, '.format(discL)
                outString += 'G: {:0.10f}, '.format(ganL)
                print outString
                if batch_ind % 10 == 0:
                    image = self.combine_images(fakeX)
                    image = image * 127.5 + 127.5
                    Image.fromarray(image.astype(np.uint8)).save(
                        osp.join(_outDirPath, '{:07d}.{:07d}.png'\
                                              .format(epoch, batch_ind)))
#                if index % 10 == 9:
#                    generator.save_weights('generator', True)
#                    discriminator.save_weights('discriminator', True)
#                sys.exit(0)

    def combine_images(self, generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[2:]
        image = np.zeros((height*shape[0], width*shape[1]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0],
                  j*shape[1]:(j+1)*shape[1]] = img[0, :, :]
        return image



BinaryBasicModel(128, 30).compille().train()
sys.exit(0)












































class BasicModel():
    def __init__(self, batch_size, num_epochs, **kwargs):
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def compille(self):
        self._init_mnist()
        self._init_generator()
        self._init_discriminator()
        self._init_gan()
        self.compiled = True
        self.G.compile(loss='binary_crossentropy', optimizer="SGD")
        self.GAN.compile(loss='binary_crossentropy',
                         optimizer=SGD(lr=5e-4, momentum=0.9, nesterov=True))
        self.D.trainable = True
        self.D.compile(loss='binary_crossentropy',
                       optimizer=SGD(lr=5e-4, momentum=0.9, nesterov=True))
        return self

    def _init_mnist(self):
        (trainX, _), _ = mnist.load_data()
        trainX = (trainX.astype(np.float32) - 127.5) / 127.5
        trainX = trainX.reshape((trainX.shape[0], 1) + trainX.shape[1:])
        self.trainX = trainX
        self.num_batches = int(self.trainX.shape[0] / self.batch_size)
        return self

    def _init_generator(self):
#        I = Input(shape=(100,))
#        x = Dense(1024, activation='sigmoid')(I)
#        x = Dense(128*7*7, activation='sigmoid')(x)
#        x = Reshape((128, 7, 7))(x)
#        x = UpSampling2D(size=(2, 2))(x)
#        x = Convolution2D(64, 5, 5, border_mode='same')(x)
#        x = Activation('sigmoid')(x)
#        x = UpSampling2D(size=(2, 2))(x)
#        x = Convolution2D(1, 5, 5, border_mode='same')(x)
#        O = Activation('sigmoid')(x)
#        self.G = Model(input=I, output=O)
#        return self
        model = Sequential()
        model.add(Dense(input_dim=100, output_dim=1024))
        model.add(Activation('tanh'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((128, 7, 7), input_shape=(128*7*7,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(64, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(1, 5, 5, border_mode='same'))
        model.add(Activation('tanh'))
        self.G = model
        return self
        

    def _init_discriminator(self):
#        I = Input(shape=(1, 28, 28))
#        x = Convolution2D(64, 5, 5, border_mode='same')(I)
#        x = Activation('tanh')(x)
#        x = MaxPooling2D(pool_size=(2, 2))(x)
#        x = Convolution2D(128, 5, 5)(x)
#        x = Activation('tanh')(x)
#        x = MaxPooling2D(pool_size=(2, 2))(x)
#        x = Flatten()(x)
#        x = Dense(1024)(x)
#        x = Activation('tanh')(x)
#        x = Dense(1)(x)
#        O = Activation('sigmoid')(x)
#        self.D = Model(input=I, output=O)
#        return self
        model = Sequential()
        model.add(Convolution2D(
                            64, 5, 5,
                            border_mode='same',
                            input_shape=(1, 28, 28)))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(128, 5, 5))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        self.D = model
        return self

    def _init_gan(self):
        model = Sequential()
        model.add(self.G)
        self.D.trainable = False
        model.add(self.D)
        self.GAN = model
        return self

    def train(self):
        if not self.compiled:
            self.compille()
        noise = np.zeros((self.batch_size, 100))
        for epoch in range(self.num_epochs):
            print 'Epoch {c:0{z}d}/{t:d}'.format(c=epoch, 
                                                 t=self.num_epochs,
                                                 z=len(str(self.num_epochs)))
            for batch_ind in range(self.num_batches):
                outString = '... {c:0{z}d}/{t:d}) '.format(c=batch_ind,
                    t=self.num_batches, z=len(str(self.num_batches)))
                for sample_ind in range(self.batch_size):
                    noise[sample_ind, :] = np.random.uniform(-1, 1, 100)
                self.D.trainable = True
                realX = self.trainX[batch_ind * self.batch_size:
                                    (batch_ind+1)*self.batch_size]
                fakeX = self.G.predict(noise, verbose=0)
                discX = np.concatenate((realX, fakeX))
                discY = [1] * self.batch_size + [0] * self.batch_size
                discL = self.D.train_on_batch(discX, discY)
                self.D.trainable = False
                for i in range(self.batch_size):
                    noise[i, :] = np.random.uniform(-1, 1, 100)
                ganL = self.GAN.train_on_batch(noise, [1] * self.batch_size)
                outString += 'D: {:0.10f}, '.format(discL)
                outString += 'G: {:0.10f}, '.format(ganL)
                print outString
                if batch_ind % 10 == 0:
                    image = self.combine_images(fakeX)
                    image = image * 127.5 + 127.5
                    Image.fromarray(image.astype(np.uint8)).save(
                        osp.join(_outDirPath, '{:07d}.{:07d}.png'\
                                              .format(epoch, batch_ind)))
#                if index % 10 == 9:
#                    generator.save_weights('generator', True)
#                    discriminator.save_weights('discriminator', True)
#                sys.exit(0)

    def combine_images(self, generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[2:]
        image = np.zeros((height*shape[0], width*shape[1]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0],
                  j*shape[1]:(j+1)*shape[1]] = img[0, :, :]
        return image



BasicModel(128, 30).compille().train()


#    def generate(self, BATCH_SIZE, nice=False):
#        generator = self.generator_model()
#        generator.compile(loss='binary_crossentropy', optimizer="SGD")
#        generator.load_weights('generator')
#        if nice:
#            discriminator = self.discriminator_model()
#            discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
#            discriminator.load_weights('discriminator')
#            noise = np.zeros((BATCH_SIZE*20, 100))
#            for i in range(BATCH_SIZE*20):
#                noise[i, :] = np.random.uniform(-1, 1, 100)
#            generated_images = generator.predict(noise, verbose=1)
#            d_pret = discriminator.predict(generated_images, verbose=1)
#            index = np.arange(0, BATCH_SIZE*20)
#            index.resize((BATCH_SIZE*20, 1))
#            pre_with_index = list(np.append(d_pret, index, axis=1))
#            pre_with_index.sort(key=lambda x: x[0], reverse=True)
#            nice_images = np.zeros((BATCH_SIZE, 1) +
#                                   (generated_images.shape[2:]), dtype=np.float32)
#            for i in range(int(BATCH_SIZE)):
#                idx = int(pre_with_index[i][1])
#                nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
#            image = self.combine_images(nice_images)
#        else:
#            noise = np.zeros((BATCH_SIZE, 100))
#            for i in range(BATCH_SIZE):
#                noise[i, :] = np.random.uniform(-1, 1, 100)
#            generated_images = generator.predict(noise, verbose=1)
#            image = self.combine_images(generated_images)
#        image = image*127.5+127.5
#        Image.fromarray(image.astype(np.uint8)).save(
#            "generated_image.png")
#



#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#def get_args():
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--mode", type=str)
#    parser.add_argument("--batch_size", type=int, default=128)
#    parser.add_argument("--nice", dest="nice", action="store_true")
#    parser.set_defaults(nice=False)
#    args = parser.parse_args()
#    return args
#
#if __name__ == "__main__":
#    args = get_args()
#    if args.mode == "train":
#        train(BATCH_SIZE=args.batch_size)
#    elif args.mode == "generate":
#        generate(BATCH_SIZE=args.batch_size, nice=args.nice)