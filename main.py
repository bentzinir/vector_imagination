import tensorflow as tf
from nn import NN
import numpy as np
import numpy.matlib
from moving_box import MovingBox
import time

batch_size = 24
lr = 0.0002
beta1 = 0.5
tv_weight = 0.05
layer_widths = [750, 450, 350]
saved_model = '/home/nir/work/git/vector_imagination/snapshots/2017-06-21-13-14-000000.sn'
train_mode = False
n_train_iters = 100000
diaplay_interval = 100
save_interval = 100
state_coord = 1

env = MovingBox(low_size=12, high_size=32, radius=3)

conv_ae = NN(batch_size, env.vector_dim, [32, 32], tv_weight, lr, beta1)

sess = tf.Session()

init_graph = tf.global_variables_initializer()

sess.run(init_graph)

# generate discriminator graph first
loss_d, apply_grads_d, grad_norm_d = conv_ae.train_discriminator()
loss_g, apply_grads_g, grad_norm_g, im_fake = conv_ae.train_generator()
loss_ae, apply_grads_ae, grad_norm_ae = conv_ae.train_decoder()

saver = tf.train.Saver()
init_graph = tf.global_variables_initializer()
sess = tf.Session()

if saved_model is None:
    sess.run(init_graph)
else:
    saver.restore(sess, saved_model)

for i in xrange(n_train_iters):

    if train_mode:
        for _ in xrange(1):
            # train discriminator
            _, positions, im_high = env.create_batch(batch_size)
            run_vals_d = sess.run(fetches=[apply_grads_d, loss_d, grad_norm_d],
                                  feed_dict={conv_ae.x: positions, conv_ae.im_expert: im_high})

        for _ in xrange(2):
            # train autoencoder
            _, positions, im_high = env.create_batch(batch_size)
            run_vals_ae = sess.run(fetches=[apply_grads_ae, loss_ae],
                                   feed_dict={conv_ae.x: positions, conv_ae.im_expert: im_high})

        for _ in xrange(2):
            # train_generator
            im_low, positions, im_high = env.create_batch(batch_size)
            run_vals_g = sess.run(fetches=[im_fake, apply_grads_g, loss_g, grad_norm_g],
                                  feed_dict={conv_ae.x: positions})

        if i % diaplay_interval == 0:
            env.update_figure(im_low[0], run_vals_g[0][0], im_high[0])
            env.update_stats({'loss_d': run_vals_d[1], 'loss_g': run_vals_g[2], 'loss_ae': run_vals_ae[1],
                              'grads_g': run_vals_g[3], 'grads_d': run_vals_d[2]})
            env.print_info_line(i, ['loss_d', 'grads_d', 'loss_g', 'grads_g', 'loss_ae'])
            fname = 'snapshots/' + time.strftime("%Y-%m-%d-%H-%M-") + ('%0.6d.sn' % i)
            saver.save(sess, fname)

    else:  # test
        im_low, positions, im_high = env.create_batch(batch_size)
        for row in xrange(12):
            for col in xrange(12):
                positions[0] = [row, col]
                im_low[0] = env.render_image(positions[0], env.r)
                x_fake = sess.run(fetches=[im_fake], feed_dict={conv_ae.x: positions})[0]
                env.update_figure(im_low[0], x_fake[0], im_high[0])
                time.sleep(0.1)
