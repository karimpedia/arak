#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:44:25 2017

@author: karim
"""

import time

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


import arak
import arak.exp


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_tgt = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_est = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = \
    tf.reduce_mean(-tf.reduce_sum(y_tgt * tf.log(y_est),
                                  reduction_indices=[1]))

cross_entropy = \
    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_est, y_tgt))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

true_positives = tf.equal(tf.argmax(y_est, 1), tf.argmax(y_tgt, 1))
accuracy = tf.reduce_mean(tf.cast(true_positives, tf.float32))


init = tf.global_variables_initializer()
sess = arak.get_tensorflow_session()
sess.run(init)


start_time = time.time()
start_epoch = -1
for i in range(100000):
    batch = mnist.train.next_batch(100)
    train_step.run(session=sess, feed_dict={x: batch[0], y_tgt: batch[1]})

    if i % 100 == 0:
        accuracy_trn = accuracy.eval(session=sess,
                                     feed_dict={x: mnist.train.images,
                                                y_tgt: mnist.train.labels})
        accuracy_vld = accuracy.eval(session=sess,
                                     feed_dict={x: mnist.test.images,
                                                y_tgt: mnist.test.labels})
        end_time = time.time()
        end_epoch = i
        run_time = '{:010.7f}'.format((end_time - start_time) /
                                      (end_epoch - start_epoch))
        start_time = end_time
        start_epoch = end_epoch
        print '... [{} seconds per epoch] batch {:04d}) '\
            'ACC (trn):{:0.5f}, ACC (vld):{:0.5f}'\
            .format(run_time, end_epoch, accuracy_trn, accuracy_vld)
