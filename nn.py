import numpy as np
import tensorflow as tf
import utils


class NN(object):
    def __init__(self, batch_size, x_dim, im_dim, tv_weight, lr=0.001):

        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, shape=[batch_size, x_dim], name='x')

        self.im_expert = tf.placeholder(tf.float32, shape=[batch_size, im_dim[0], im_dim[1], 1], name='im_expert')

        self.x_dim = x_dim

        self.tv_weight = tv_weight

        self.solver_params = {
            'lr': lr,
            'weight_decay_rate': 0.000001,
        }

    def _affine(self, name, x, in_filters, out_filters):
        with tf.variable_scope(name):
            n = in_filters * out_filters
            weights = tf.get_variable('_weight', [in_filters, out_filters], tf.float32,
                                      initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            bias = tf.get_variable('bias', [out_filters], tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))

            h = tf.matmul(x, weights)
            return tf.nn.bias_add(h, bias)
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

    def conv2d(self, input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

            return conv

    def _conv_transpose(self, name, x, filter_size, in_filters, out_filters):
        n = filter_size * filter_size * in_filters
        x_dims = np.asarray(x.get_shape().as_list())
        output_shape = [self.batch_size, x_dims[1]*2, x_dims[2]*2, out_filters]
        with tf.variable_scope(name):
            kernel = tf.get_variable(name + '_weight', [filter_size, filter_size, out_filters, in_filters], tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))

            bias = tf.get_variable(name + '_bias', [out_filters], tf.float32,
                                   initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))

            h = tf.nn.conv2d_transpose(x, kernel, output_shape=output_shape, strides=[1, 2, 2, 1], padding='SAME', data_format='NHWC')
            return tf.nn.bias_add(h, bias, data_format='NHWC')

    def _decay(self, scope):
        """L2 weight decay loss."""
        costs = []

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        for var in train_vars:
            if var.op.name.find(r'_weight') > 0:
                costs.append(self.solver_params['weight_decay_rate'] * tf.nn.l2_loss(var))

        return self.solver_params['weight_decay_rate'] * tf.add_n(costs)

    def generator(self, reuse=False):

        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            x = self.x

            x = self._affine(name='affine1', x=x, in_filters=self.x_dim, out_filters=4*4*1024)

            # x = tf.contrib.layers.batch_norm(x, scale=True)

            # x = tf.nn.relu(x)

            # x = self._affine(name='affine2', x=x, in_filters=100, out_filters=256)
            #
            #
            # x = tf.contrib.layers.batch_norm(x, scale=True)

            # x = tf.nn.relu(x)

            # x = self._affine(name='affine3', x=x, in_filters=256, out_filters=4*4*1024)

            # x = tf.contrib.layers.batch_norm(x, scale=True)

            x = tf.reshape(x, [self.batch_size, 4, 4, 1024])

            x = self._conv_transpose(name='deconv1', x=x, filter_size=3, in_filters=1024, out_filters=512)

            x = tf.contrib.layers.batch_norm(x, scale=True)

            x = tf.nn.relu(x)

            x = self._conv_transpose(name='deconv2', x=x, filter_size=3, in_filters=512, out_filters=256)

            x = tf.contrib.layers.batch_norm(x, scale=True)

            x = tf.nn.relu(x)

            x = self._conv_transpose(name='deconv3', x=x, filter_size=3, in_filters=256, out_filters=1)

            # x = tf.contrib.layers.batch_norm(x, scale=True)
            #
            # debug_x = x
            #
            # x = self._conv_transpose(name='deconv4', x=x, filter_size=3, in_filters=64, out_filters=1)

            im = tf.nn.tanh(x)

            return im

    def decoder(self, im, reuse=False):

        with tf.variable_scope("decoder") as scope:
            if reuse:
                scope.reuse_variables()

            # x = self._conv_pool_relu(name='conv1', x=im, filter_size=3, in_filters=1, out_filters=16, pool_kernel=2, pool_stride=2)
            x = self.conv2d(im, 16, name='conv1')

            x = tf.contrib.layers.batch_norm(x, scale=True)

            # x = self._conv_pool_relu(name='conv2', x=x, filter_size=3, in_filters=16, out_filters=64, pool_kernel=2, pool_stride=2)
            x = self.conv2d(im, 64, name='conv2')

            x = tf.contrib.layers.batch_norm(x, scale=True)

            # x = self._conv_pool_relu(name='conv3', x=x, filter_size=3, in_filters=64, out_filters=256, pool_kernel=2, pool_stride=2)
            #
            # x = tf.contrib.layers.batch_norm(x, scale=True)
            #
            # x = self._conv_pool_relu(name='conv4', x=x, filter_size=3, in_filters=256, out_filters=512, pool_kernel=2, pool_stride=2)

            x = tf.reshape(x, [self.batch_size, -1])

            n_features = x.get_shape().as_list()[1]

            x = self._affine(name='affine4', x=x, in_filters=n_features, out_filters=100)

            x = tf.contrib.layers.batch_norm(x, scale=True)

            x = tf.nn.relu(x)

            x = self._affine(name='affine5', x=x, in_filters=100, out_filters=self.x_dim)

            return x

    def discriminator(self, im, reuse=False):

        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # d = self._conv_pool_relu(name='conv1', x=im, filter_size=3, in_filters=1, out_filters=16, pool_kernel=2, pool_stride=2)
            d = self.conv2d(im, 16, name='conv1')

            d = tf.contrib.layers.batch_norm(d, scale=True)

            # d = self._conv_pool_relu(name='conv2', x=d, filter_size=3, in_filters=16, out_filters=64, pool_kernel=2, pool_stride=2)
            d = self.conv2d(d, 64, name='conv2')

            d = tf.contrib.layers.batch_norm(d, scale=True)

            # d = self._conv_pool_relu(name='conv3', x=d, filter_size=3, in_filters=64, out_filters=256, pool_kernel=2, pool_stride=2)
            d = self.conv2d(d, 128, name='conv3')

            # d = tf.contrib.layers.batch_norm(d, scale=True)
            #
            # d = self._conv_pool_relu(name='conv4', x=d, filter_size=3, in_filters=256, out_filters=512, pool_kernel=2, pool_stride=2)

            d = tf.reshape(d, [self.batch_size, -1])

            n_features = d.get_shape().as_list()[1]

            d = self._affine(name='affine4', x=d, in_filters=n_features, out_filters=100)

            d = tf.contrib.layers.batch_norm(d, scale=True)

            d = tf.nn.relu(d)

            d = self._affine(name='affine5', x=d, in_filters=100, out_filters=1)

        return d

    def backward(self, loss, scopes):
        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'])

        train_vars = []
        for scope in scopes:
            train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=train_vars)

        g_norm, w_norm = utils.compute_mean_abs_norm(grads_and_vars)

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads, g_norm, w_norm

    def train_discriminator(self):

        im_fake = self.generator(reuse=False)

        d_real = self.discriminator(self.im_expert, reuse=False)

        d_fake = self.discriminator(im_fake, reuse=True)

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))

        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))

        loss = d_loss_real + d_loss_fake

        # loss += self._decay("discriminator")

        return loss, self.backward(loss, ["discriminator"])

    def train_generator(self):

        im_fake= self.generator(reuse=True)

        # loss = tf.reduce_mean(tf.square(self.im_expert - im_fake))

        d_g = self.discriminator(im_fake, reuse=True)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g, labels=tf.ones_like(d_g)))

        # x_rec = self.decoder(im_fake)

        # rec_loss = tf.nn.l2_loss(x_rec - self.x)

        # tv_loss = self.tv_weight * tf.reduce_mean(tf.image.total_variation(im_fake))

        # loss += rec_loss + tv_loss

        # loss += self._decay("decoder")

        # loss += self._decay("generator")

        return loss, self.backward(loss, ["generator"]), im_fake

    # def train_decoder(self):
    #
    #     im_fake = self.generator(reuse=True)
    #
    #     x_rec = self.decoder(im_fake)
    #
    #     rec_loss = tf.nn.l2_loss(x_rec - self.x)
    #
    #     tv_loss = self.tv_weight * tf.reduce_mean(tf.image.total_variation(im_fake))
    #
    #     loss = rec_loss + tv_loss
    #
    #     loss += self._decay("decoder")
    #
    #     return loss, self.backward(loss, ["decoder", "generator"])

