import numpy as np
import tensorflow as tf
import utils


class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
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

class NN(object):
    def __init__(self, batch_size, x_dim, im_dim, tv_weight, lr, beta1):

        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, shape=[batch_size, x_dim], name='x')

        self.im_expert = tf.placeholder(tf.float32, shape=[batch_size, im_dim[0], im_dim[1], 1], name='im_expert')

        self.output_height = im_dim[0]

        self.output_width = im_dim[1]

        self.gf_dim = 32
        self.df_dim = 32

        self.gfc_dim = 1024
        self.dfc_dim = 1024

        self.x_dim = x_dim

        self.tv_weight = tv_weight

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.g_opt = None

        self.solver_params = {
            'lr': lr,
            'weight_decay_rate': 0.000001,
            'beta1': beta1,
        }

    # def _affine(self, name, x, in_filters, out_filters):
    #     with tf.variable_scope(name):
    #         n = in_filters * out_filters
    #         weights = tf.get_variable('_weight', [in_filters, out_filters], tf.float32,
    #                                   initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
    #         bias = tf.get_variable('bias', [out_filters], tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
    #
    #         h = tf.matmul(x, weights)
    #         return tf.nn.bias_add(h, bias)
    #
    # def _conv(self, name, x, filter_size, in_filters, out_filters):
    #     """Convolution."""
    #     with tf.variable_scope(name):
    #         n = filter_size * filter_size * out_filters
    #         kernel = tf.get_variable(name + '_weight', [filter_size, filter_size, in_filters, out_filters], tf.float32,
    #                                  initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
    #
    #         bias = tf.get_variable(name + '_bias', [out_filters], tf.float32,
    #                                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
    #
    #         h = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME', data_format='NHWC')
    #         return tf.nn.bias_add(h, bias, data_format='NHWC')
    #
    # def _conv_pool_relu(self, name, x, filter_size, in_filters, out_filters, pool_kernel, pool_stride, relu=True):
    #     x = self._conv(name + '_conv', x, filter_size, in_filters, out_filters)
    #     # if pool_kernel is not None:
    #     #     x = tf.nn.max_pool(x, [1, pool_kernel, pool_kernel, 1], [1, pool_stride, pool_stride, 1],
    #     #                        padding='SAME', data_format='NHWC', name=name + '_maxpool')
    #     if relu is True:
    #         x = tf.nn.relu(x)
    #     return x

    def linear(self, input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
        shape = input_.get_shape().as_list()

        with tf.variable_scope(scope or "Linear"):
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable("bias", [output_size],
                                   initializer=tf.constant_initializer(bias_start))
            if with_w:
                return tf.matmul(input_, matrix) + bias, matrix, bias
            else:
                return tf.matmul(input_, matrix) + bias

    def conv2d(self, input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

            return conv

    # def _conv_transpose(self, name, x, filter_size, in_filters, out_filters):
    #     n = filter_size * filter_size * in_filters
    #     x_dims = np.asarray(x.get_shape().as_list())
    #     output_shape = [self.batch_size, x_dims[1]*2, x_dims[2]*2, out_filters]
    #     with tf.variable_scope(name):
    #         kernel = tf.get_variable(name + '_weight', [filter_size, filter_size, out_filters, in_filters], tf.float32,
    #                                  initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
    #
    #         bias = tf.get_variable(name + '_bias', [out_filters], tf.float32,
    #                                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
    #
    #         h = tf.nn.conv2d_transpose(x, kernel, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')
    #         return tf.nn.bias_add(h, bias, data_format='NHWC')

    def deconv2d(self, input_, output_shape,
                 k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                 name="deconv2d", with_w=False):
        with tf.variable_scope(name):
            # filter : [height, width, output_channels, in_channels]
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

            try:
                # deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                #                                 strides=[1, d_h, d_w, 1])

                deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                                strides=[1, d_h, d_w, 1])

            # Support for verisons of TensorFlow before 0.7.0
            except AttributeError:
                deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])

            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

            if with_w:
                return deconv
            else:
                return deconv

    def _decay(self, scope):
        """L2 weight decay loss."""
        costs = []

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        for var in train_vars:
            if var.op.name.find(r'_weight') > 0:
                costs.append(self.solver_params['weight_decay_rate'] * tf.nn.l2_loss(var))

        return self.solver_params['weight_decay_rate'] * tf.add_n(costs)

    def conv_out_size_same(self, size, stride):
        import math
        return int(math.ceil(float(size) / float(stride)))

    def generator(self, reuse=False):

        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2)

            z = self.x
            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = self.linear(
                z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1 = self.deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2 = self.deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = self.deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4 = self.deconv2d(
                h3, [self.batch_size, s_h, s_w, 1], name='g_h4', with_w=True)

            im = tf.nn.tanh(h4)

            return im

    def decoder(self, im, reuse=False):

        with tf.variable_scope("decoder") as scope:
            if reuse:
                scope.reuse_variables()

            # x = self._conv_pool_relu(name='conv1', x=im, filter_size=3, in_filters=1, out_filters=16, pool_kernel=2, pool_stride=2)
            x = self.conv2d(im, 16, name='conv1')

            x = tf.contrib.layers.batch_norm(x, scale=True)

            # x = self._conv_pool_relu(name='conv2', x=x, filter_size=3, in_filters=16, out_filters=64, pool_kernel=2, pool_stride=2)
            x = self.conv2d(x, 64, name='conv2')

            x = tf.contrib.layers.batch_norm(x, scale=True)

            # x = self._conv_pool_relu(name='conv3', x=x, filter_size=3, in_filters=64, out_filters=256, pool_kernel=2, pool_stride=2)
            #
            # x = tf.contrib.layers.batch_norm(x, scale=True)
            #
            # x = self._conv_pool_relu(name='conv4', x=x, filter_size=3, in_filters=256, out_filters=512, pool_kernel=2, pool_stride=2)

            x = tf.reshape(x, [self.batch_size, -1])

            n_features = x.get_shape().as_list()[1]

            # x = self._affine(x, name='affine4', x=x, in_filters=n_features, out_filters=100)

            x = self.linear(x, 100, 'affine4', with_w=False)

            x = tf.contrib.layers.batch_norm(x, scale=True)

            x = tf.nn.relu(x)

            # x = self._affine(name='affine5', x=x, in_filters=100, out_filters=)
            x = self.linear(x, self.x_dim, 'affine5', with_w=False)

            return x

    def lrelu(self, x, leak=0.2):
        return tf.maximum(x, leak * x)

    def discriminator(self, im, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = self.lrelu(self.conv2d(im, self.df_dim, name='d_h0_conv'))
            h1 = self.lrelu(self.d_bn1(self.conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = self.lrelu(self.d_bn2(self.conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = self.lrelu(self.d_bn3(self.conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = self.linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

            return h4

    def backward(self, loss, scopes, opt=None):
        # create an optimizer
        if opt is None:
            opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'], beta1=self.solver_params['beta1'])

        train_vars = []
        for scope in scopes:
            train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=train_vars)

        g_norm, w_norm = utils.compute_mean_abs_norm(grads_and_vars)

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads, g_norm, w_norm, opt

    def train_discriminator(self):

        im_fake = self.generator(reuse=False)

        d_real = self.discriminator(self.im_expert, reuse=False)

        d_fake = self.discriminator(im_fake, reuse=True)

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))

        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))

        loss = d_loss_real + d_loss_fake

        loss += self._decay("discriminator")

        apply_grads, grad_norm, _, _ = self.backward(loss, ["discriminator"])

        return loss, apply_grads, grad_norm

    def train_generator(self):

        im_fake = self.generator(reuse=True)

        # loss = tf.reduce_mean(tf.square(self.im_expert - im_fake))

        d_g = self.discriminator(im_fake, reuse=True)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g, labels=tf.ones_like(d_g)))

        apply_grads, grad_norm, _, self.g_opt = self.backward(loss, ["generator"], self.g_opt)

        return loss, apply_grads, grad_norm, im_fake

    def train_decoder(self):

        im_fake = self.generator(reuse=True)

        x_rec = self.decoder(im_fake)

        rec_loss = tf.reduce_mean(tf.square(x_rec - self.x))

        tv_loss = self.tv_weight * tf.reduce_mean(tf.image.total_variation(im_fake))

        loss = rec_loss + tv_loss

        # loss += self._decay("decoder")

        apply_grads, grad_norm, _, self.g_opt = self.backward(loss, ["generator", "decoder"], self.g_opt)

        return loss, apply_grads, grad_norm

