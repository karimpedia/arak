#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:44:25 2017

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
# makedirpath(_outDirPath)
# -----------------------------------------------------------------------------
# =============================================================================


# ======================================================== Flags and parameters
# -----------------------------------------------------------------------------
# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10
# -----------------------------------------------------------------------------
# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
BATCH_SIZE = 100
FAKE_DATA = False  # If true, use fake data for unit tests
INPUT_DATA_DIR = '/tmp/tensorflow/mnist/logs/fully_connected_feed'
HIDDEN_1 = 128
HIDDEN_2 = 32
LEARNING_RATE = 0.01
LOG_DIR = _outDirPath
MAX_STEPS = 200000
# -----------------------------------------------------------------------------
# =============================================================================


def inference(images, hidden1_units=HIDDEN_1, hidden2_units=HIDDEN_2):
    with tf.name_scope('hidden1'):
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        weights = \
            tf.Variable(
                tf.truncated_normal(
                    [IMAGE_PIXELS, hidden1_units],
                    stddev=1.0/math.sqrt(float(IMAGE_PIXELS))),
                name='weights')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    with tf.name_scope('hidden2'):
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        weights = \
            tf.Variable(
                tf.truncated_normal(
                    [hidden1_units, hidden2_units],
                    stddev=1.0/math.sqrt(float(hidden1_units))),
                name='weights')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    with tf.name_scope('readout'):
        biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
        weights = \
            tf.Variable(
                tf.truncated_normal(
                    [hidden2_units, NUM_CLASSES],
                    stddev=1.0/math.sqrt(float(hidden1_units))),
                name='weights')
        logits = tf.matmul(hidden2, weights) + biases
    return logits


def get_loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def placeholder_inputs(batch_size):
    images_placeholder = \
        tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
    labels_placeholder = \
        tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(BATCH_SIZE, FAKE_DATA)
    feed_dict = {images_pl: images_feed, labels_pl: labels_feed}
    return feed_dict


def evaluate(session, eval_correct, images_pl, labels_pl, data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_pl, labels_pl)
        true_count += session.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print '... Evaluation: {c:05d}/{t:05d} (acc:{acc:05.3f})'\
        .format(z=len(str(num_examples)), t=num_examples,
                c=true_count, acc=precision)


def train():
    data_sets = read_data_sets(INPUT_DATA_DIR, FAKE_DATA)
    with tf.Graph().as_default():
        images_pl, labels_pl = placeholder_inputs(BATCH_SIZE)
        fLogits = inference(images_pl)
        fLoss = get_loss(fLogits, labels_pl)
        train_op = training(fLoss, LEARNING_RATE)
        eval_correct = evaluation(fLogits, labels_pl)
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        sess.run(init)
        for step in xrange(MAX_STEPS):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets.train, images_pl, labels_pl)
            _, loss_value = sess.run([train_op, fLoss], feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 100 == 0:
                print 'Step {}) Loss = {:0.3f} ({:0.3f} seconds)'\
                    .format(step, loss_value, duration)
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join(LOG_DIR, 'MODEL.CKPT')
                saver.save(sess, checkpoint_file, global_step=step)
                print 'Training data eval:'
                evaluate(sess, eval_correct, images_pl, labels_pl, data_sets.train)
                print 'Validation data eval:'
                evaluate(sess, eval_correct, images_pl, labels_pl, data_sets.validation)
                print 'Testing data eval:'
                evaluate(sess, eval_correct, images_pl, labels_pl, data_sets.test)


def main(_):
    if tf.gfile.Exists(LOG_DIR):
        tf.gfile.DeleteRecursively(LOG_DIR)
    tf.gfile.MakeDirs(LOG_DIR)
    train()


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])