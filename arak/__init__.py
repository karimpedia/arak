#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 21:11:51 2017

@author: karim
"""

import os


ENV_TF_GPU_MEM_FRAC = ('GPU_MEMFRAC', 0.333)
ENV_OMP_NUM_THREADS = ('OMP_NUM_THREADS', None)


def get_tensorflow_session(gpu_memfrac=None):
    import tensorflow as tf
    if not gpu_memfrac:
        gpu_memfrac = os.environ.get(ENV_TF_GPU_MEM_FRAC[0]) \
            if os.environ.get(ENV_TF_GPU_MEM_FRAC[0]) is not None \
            else ENV_TF_GPU_MEM_FRAC[1]
    num_threads = os.environ.get(ENV_OMP_NUM_THREADS[0]) \
        if os.environ.get(ENV_OMP_NUM_THREADS[0]) is not None \
        else ENV_OMP_NUM_THREADS[1]
    gpu_options = \
        tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=gpu_memfrac)
    if num_threads:
        cSession = \
            tf.Session(
               config=tf.ConfigProto(
                    gpu_options=gpu_options,
                    intra_op_parallelism_threads=num_threads))
    else:
        cSession = \
            tf.Session(
               config=tf.ConfigProto(
                    gpu_options=gpu_options))
    return cSession


def default():
    pass


default()
