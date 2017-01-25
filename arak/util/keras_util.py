#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:42:09 2017

@author: karim
"""


def set_trainable(model, trainable):
    model.trainable = trainable
    for x in model.layers:
        x.trainable = trainable


def freeze_model(model):
    set_trainable(model, False)


def unfreeze_model(model):
    set_trainable(model, True)
