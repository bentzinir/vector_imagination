import numpy as np
import tensorflow as tf


class NN(object):
    def __init__(self, batch_size, x_dim, lr=0.001):

        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, shape=[batch_size, x_dim], name='x')

        self.x_dim = x_dim

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

    def _conv(self, name, x, filter_size, in_filters, out_filters):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(name + '_weight', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                     initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))

            bias = tf.get_variable(name + '_bias', [out_filters], tf.float32,
                                   initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))

            h = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME', data_format='NHWC')
            return tf.nn.bias_add(h, bias, data_format='NHWC')

    def _conv_pool_relu(self, name, x, filter_size, in_filters, out_filters, pool_kernel, pool_stride, relu=True):
        x = self._conv(name + '_conv', x, filter_size, in_filters, out_filters)
        if pool_kernel is not None:
            x = tf.nn.max_pool(x, [1, pool_kernel, pool_kernel, 1], [1, pool_stride, pool_stride, 1],
                               padding='SAME', data_format='NHWC', name=name + '_maxpool')
        if relu is True:
            x = tf.nn.relu(x)
        return x

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

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'_weight') > 0:
                costs.append(self.solver_params['weight_decay_rate'] * tf.nn.l2_loss(var))

        return self.solver_params['weight_decay_rate'] * tf.add_n(costs)

    def forward(self):

        x = self.x

        x = self._affine(name='affine1', x=x, in_filters=self.x_dim, out_filters=100)

        x = tf.nn.relu(x)

        x = self._affine(name='affine2', x=x, in_filters=100, out_filters=256)

        x = tf.nn.relu(x)

        x = self._affine(name='affine3', x=x, in_filters=256, out_filters=4*4*1024)

        x = tf.reshape(x, [self.batch_size, 4, 4, 1024])

        x = self._conv_transpose(name='deconv1', x=x, filter_size=3, in_filters=1024, out_filters=512)

        x = tf.nn.relu(x)

        x = self._conv_transpose(name='deconv2', x=x, filter_size=3, in_filters=512, out_filters=256)

        x = tf.nn.relu(x)

        x = self._conv_transpose(name='deconv3', x=x, filter_size=3, in_filters=256, out_filters=64)

        x = tf.nn.relu(x)

        im = self._conv_transpose(name='deconv4', x=x, filter_size=3, in_filters=64, out_filters=1)

        x = self._conv_pool_relu(name='conv1', x=im, filter_size=3, in_filters=1, out_filters=16, pool_kernel=2, pool_stride=2)

        x = self._conv_pool_relu(name='conv2', x=x, filter_size=3, in_filters=16, out_filters=64, pool_kernel=2, pool_stride=2)

        x = self._conv_pool_relu(name='conv3', x=x, filter_size=3, in_filters=64, out_filters=256, pool_kernel=2, pool_stride=2)

        x = self._conv_pool_relu(name='conv4', x=x, filter_size=3, in_filters=256, out_filters=512, pool_kernel=2, pool_stride=2)

        x = tf.reshape(x, [self.batch_size, -1])

        n_features = x.get_shape().as_list()[1]

        x = self._affine(name='affine4', x=x, in_filters=n_features, out_filters=100)

        x = tf.nn.relu(x)

        x = self._affine(name='affine5', x=x, in_filters=100, out_filters=self.x_dim)

        return x, im

    def backward(self, loss):
        # create an optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.solver_params['lr'])

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=tf.trainable_variables())

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads

    def train(self):

        x_r, im = self.forward()

        loss = tf.nn.l2_loss(self.x - x_r)

        loss += 0.01 * tf.reduce_mean(tf.image.total_variation(im))

        loss += self._decay()

        return loss, im, self.backward(loss)

