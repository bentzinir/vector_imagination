import tensorflow as tf
import utils
import params


class NN(object):
    def __init__(self, x_dim, im_dim):

        self.output_height = im_dim[0]

        self.output_width = im_dim[1]

        self.gf_dim = 8
        self.df_dim = 32

        self.gfc_dim = 1024
        self.dfc_dim = 1024

        self.x_dim = x_dim

        self.d_bn1 = utils.batch_norm(name='d_bn1')
        self.d_bn2 = utils.batch_norm(name='d_bn2')
        self.d_bn3 = utils.batch_norm(name='d_bn3')

        self.g_opt = None

    def linear(self, input_, output_size, scope=None, stddev=0.02, bias_start=0.0):
        shape = input_.get_shape().as_list()
        with tf.variable_scope(scope or "Linear"):
            w = tf.get_variable("weight", [shape[1], output_size], tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
            b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input_, w) + b

    def conv2d(self, input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('weight', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            b = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            output = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
            return tf.nn.bias_add(output, b)

    def deconv2d(self, input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('weight', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            b = tf.get_variable('bias', [output_shape[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
            output = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
        return tf.nn.bias_add(output, b)

    def _decay(self, scope):
        costs = []
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope):
            if var.op.name.find(r'weight') > 0:
                costs.append(params.weight_decay_rate * tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def deconv_res_block(self, name, x, h, w, cout):
        with tf.variable_scope(name):
            bn1 = utils.batch_norm(name=name + 'bn1')
            x = self.deconv2d(x, [params.bs, h, w, cout], k_h=3, k_w=3, name=name + '.1')
            x = tf.nn.relu(bn1(x))
            bn2 = utils.batch_norm(name=name + 'bn2')
            x = self.conv2d(x, cout, d_h=1, d_w=1, name='conv') + x
            x = tf.nn.relu(bn2(x))
        return x

    def conv_relu_block(self, name, x, cout):
        with tf.variable_scope(name):
            x = self.conv2d(x, cout, name=name)
            bn = utils.batch_norm(name=name + 'bn')
            x = utils.lrelu(bn(x))
        return x

    def generator(self, x, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = utils.conv_out_size_same(s_h, 2), utils.conv_out_size_same(s_w, 2)
            s_h4, s_w4 = utils.conv_out_size_same(s_h2, 2), utils.conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = utils.conv_out_size_same(s_h4, 2), utils.conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = utils.conv_out_size_same(s_h8, 2), utils.conv_out_size_same(s_w8, 2)

            if params.synthetic:
                z = tf.random_uniform((params.bs, params.z_dim), minval=-1, maxval=1)
            else:
                z = x

            h = self.linear(z, self.gf_dim * 4 * s_h16 * s_w16, 'affine0')
            h = tf.nn.relu(h)
            h = self.linear(h, self.gf_dim * 8 * s_h16 * s_w16, 'affine1')
            h = tf.nn.relu(tf.reshape(h, [-1, s_h16, s_w16, self.gf_dim * 8]))
            h = self.deconv_res_block('deconv0', h, s_h8, s_w8, self.gf_dim * 4)
            h = self.deconv_res_block('deconv1', h, s_h4, s_w4, self.gf_dim * 2)
            h = self.deconv_res_block('deconv2', h, s_h2, s_w2, self.gf_dim * 1)
            h = self.deconv2d(h, [params.bs, s_h, s_w, 1], k_h=3, k_w=3, name='g_h4')
            h = tf.nn.relu(h)
            return h

    def decoder(self, im, reuse=False):
        with tf.variable_scope("decoder") as scope:
            if reuse:
                scope.reuse_variables()
            h = self.conv_relu_block('conv0', im, self.df_dim)
            h = self.conv_relu_block('conv1', h, self.df_dim * 2)
            h = self.conv_relu_block('conv2', h, self.df_dim * 4)
            h = self.conv_relu_block('conv3', h, self.df_dim * 8)
            h = self.linear(tf.reshape(h, [params.bs, -1]), self.x_dim, 'de_h4_lin')
            return h

    def discriminator(self, im, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h = self.conv_relu_block('conv0', im, self.df_dim)
            h = self.conv_relu_block('conv1', h, self.df_dim * 2)
            h = self.conv_relu_block('conv2', h, self.df_dim * 4)
            feats = h
            h = self.conv_relu_block('conv3', h, self.df_dim * 8)
            h = self.linear(tf.reshape(h, [params.bs, -1]), self.x_dim, 'de_h4_lin')
            return h, tf.reshape(feats, [params.bs, -1])

    def backward(self, loss, scopes, opt=None):
        # create an optimizer
        if opt is None:
            opt = tf.train.AdamOptimizer(learning_rate=params.lr, beta1=params.beta1)

        train_vars = []
        for scope in scopes:
            train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        # compute the gradients for a list of variables
        grads_and_vars = opt.compute_gradients(loss=loss, var_list=train_vars)

        g_norm, w_norm = utils.compute_mean_abs_norm(grads_and_vars)

        # apply the gradient
        apply_grads = opt.apply_gradients(grads_and_vars)

        return apply_grads, g_norm, w_norm, opt

    def train_generator(self, x):

        # classic gan loss
        fake_im = self.generator(x, reuse=True)

        d_g, disc_feats = self.discriminator(fake_im, reuse=True)

        gan_loss = params.gan_weight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_g, labels=tf.ones_like(d_g)))

        # order loss
        # x_dists = utils.distance_mat(x)
        #
        # feats_dist = utils.distance_mat(disc_feats)

        # x_order = x_dists / tf.reduce_max(x_dists)
        #
        # feats_order = feats_dist / tf.reduce_max(feats_dist)

        # order_loss = params.order_weight * tf.reduce_mean(tf.nn.l2_loss(x_order - feats_order))

        loss = gan_loss# + order_loss

        apply_grads, grad_norm, _, self.g_opt = self.backward(loss, ["generator"], self.g_opt)

        return [gan_loss], apply_grads, grad_norm, fake_im

    def train_discriminator(self, x, ref_im):

        im_fake = self.generator(x, reuse=False)

        d_real, _ = self.discriminator(ref_im, reuse=False)

        d_fake, _ = self.discriminator(im_fake, reuse=True)

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real)))

        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))

        loss = d_loss_real + d_loss_fake

        loss += self._decay("discriminator")

        apply_grads, grad_norm, _, _ = self.backward(loss, ["discriminator"])

        return loss, apply_grads, grad_norm

    def train_decoder(self, x, x_im):

        im_fake = self.generator(x, reuse=True)

        _, disc_feats = self.discriminator(im_fake, reuse=True)

        x_dists = utils.distance_mat(x)

        feats_dist = utils.distance_mat(disc_feats)

        x_order = x_dists / tf.reduce_max(x_dists)

        feats_order = feats_dist / tf.reduce_max(feats_dist)

        order_loss = params.order_weight * tf.reduce_mean(tf.nn.l2_loss(x_order - feats_order))

        # x_rec = self.decoder(im_fake)
        #
        # rec_loss = tf.reduce_mean(tf.square(x_rec - x))
        #
        # tv_loss = params.tv_weight * tf.reduce_mean(tf.image.total_variation(im_fake))
        #
        # supervised_loss = tf.reduce_mean(tf.square(x_im - im_fake))

        loss = order_loss

        # loss += self._decay("decoder")

        apply_grads, grad_norm, _, self.g_opt = self.backward(loss, ["generator", "decoder"], self.g_opt)

        return [loss], im_fake, apply_grads, grad_norm

