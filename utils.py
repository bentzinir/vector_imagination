import tensorflow as tf
import numpy as np
import math
import params
from tqdm import tqdm


def compute_mean_abs_norm(grads_and_vars):
    tot_grad = 0
    tot_w = 0
    N = len(grads_and_vars)

    for g, w in grads_and_vars:
        tot_grad += tf.reduce_mean(tf.abs(g))
        tot_w += tf.reduce_mean(tf.abs(w))

    return tot_grad/N, tot_w/N


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def euclidean_dist(u, v):
    return tf.nn.l2_loss(u - v)


def distance_mat(x):
    bs = x.get_shape().as_list()[0]
    m1 = tf.tile(tf.expand_dims(x, axis=1), [1, bs, 1])
    m2 = tf.tile(tf.expand_dims(x, axis=0), [bs, 1, 1])
    return tf.sqrt(tf.reduce_sum((m1 - m2)**2, axis=2) + 1e-4)


def square_images_distance(images):
    bs = params.batch_size

    xx = tf.tile(tf.expand_dims(tf.range(32, dtype=tf.float32), 0), [32, 1])
    yy = tf.tile(tf.expand_dims(tf.range(32, dtype=tf.float32), 1), [1, 32])

    xxx = tf.expand_dims(tf.tile(tf.expand_dims(xx, 0), [bs, 1, 1]), 3)
    yyy = tf.expand_dims(tf.tile(tf.expand_dims(yy, 0), [bs, 1, 1]), 3)

    x = tf.reduce_mean(images * xxx, axis=[1, 2, 3])
    y = tf.reduce_mean(images * yyy, axis=[1, 2, 3])

    mean_pos = tf.stack([y, x], axis=1)

    return distance_mat(mean_pos), mean_pos


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)
