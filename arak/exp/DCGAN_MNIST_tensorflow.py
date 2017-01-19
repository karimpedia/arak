#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 02:03:51 2017

@author: karim
"""


import math
import os
import os.path as osp
import sys
import time

import numpy as np
import tensorflow as tf


from datetime import datetime


from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import arak
import arak.exp

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

