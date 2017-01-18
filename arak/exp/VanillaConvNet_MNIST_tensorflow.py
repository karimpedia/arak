#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 21:03:49 2017

@author: karim
"""

import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


import arak
import arak.exp


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev, seed=1)
    return tf.Variable(initial)


def bias_variable(shape, bias=0.1):
    initial = tf.constant(bias, shape=shape)
    return tf.Variable(initial)


def conv_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y_tgt = tf.placeholder(tf.float32, [None, 10])
keep_prob_1 = tf.placeholder(tf.float32)

X_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv_2d(X_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv_2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob_1, seed=1)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = \
    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_tgt))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

true_positives = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_tgt, 1))

accuracy = tf.reduce_mean(tf.cast(true_positives, tf.float32))


init = tf.global_variables_initializer()
sess = arak.get_tensorflow_session()
sess.run(init)


start_time = time.time()
start_epoch = -1
for i in range(20001):
    batch = mnist.train.next_batch(50)
    train_step.run(session=sess, feed_dict={x: batch[0],
                                            y_tgt: batch[1],
                                            keep_prob_1: 0.77})

    if i % 100 == 0:
        accuracy_trn = accuracy.eval(session=sess,
                                     feed_dict={x: batch[0],
                                                y_tgt: batch[1],
                                                keep_prob_1: 1.0})
        accuracy_vld = accuracy.eval(session=sess,
                                     feed_dict={x: mnist.test.images,
                                                y_tgt: mnist.test.labels,
                                                keep_prob_1: 1.0})
        end_time = time.time()
        end_epoch = i
        run_time = '{:010.7f}'.format((end_time - start_time) /
                                      (end_epoch - start_epoch))
        start_time = end_time
        start_epoch = end_epoch
        print '... [{} seconds per epoch] batch {:04d}) '\
            'ACC (trn):{:0.5f}, ACC (vld):{:0.5f}'\
            .format(run_time, end_epoch, accuracy_trn, accuracy_vld)
