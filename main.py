import tensorflow as tf
from nn import NN
from moving_box import MovingBox

batch_size = 24
lr = 0.002
tv_weight = 0.05
layer_widths = [750, 450, 350]
saved_model = None
train_mode = True
n_train_iters = 100000
diaplay_interval = 10
state_coord = 1

env = MovingBox(im_size=12, radius=3)

conv_ae = NN(batch_size, env.vector_dim, [64, 64], tv_weight, lr)

sess = tf.Session()

init_graph = tf.global_variables_initializer()

sess.run(init_graph)

# generate discriminator graph first
loss_d, apply_grads_d = conv_ae.train_discriminator()
loss_g, apply_grads_g, im_fake = conv_ae.train_generator()
# loss_ae, apply_grads_ae = conv_ae.train_decoder()

saver = tf.train.Saver()
init_graph = tf.global_variables_initializer()
sess = tf.Session()

if saved_model is None:
    sess.run(init_graph)
else:
    saver.restore(sess, saved_model)

for i in xrange(n_train_iters):
    if train_mode:

        im_low, positions, im_high = env.create_batch(batch_size)

        # train generator
        run_vals_g = sess.run(fetches=[apply_grads_g, loss_g, im_fake],
                            feed_dict={conv_ae.x: positions})

        # train discriminator
        run_vals_d = sess.run(fetches=[apply_grads_d, loss_d],
                            feed_dict={conv_ae.x: positions, conv_ae.im_expert: im_high})

        # train autoencoder
        # _, positions, im_high = env.create_batch(batch_size)
        # run_vals_ae = sess.run(fetches=[apply_grads_ae, loss_ae],
        #                     feed_dict={conv_ae.x: positions, conv_ae.im_expert: im_high})

        if i % diaplay_interval == 0:
            sample_low_im = im_low[0]
            x_recon_sample = run_vals_g[2][0, :, :, 0]
            sample_high_im = im_high[0][:, :, 0]
            env.update_stats({'loss_d': run_vals_d[1], 'loss_g': run_vals_g[1]})
            env.print_info_line(i, ['loss_d', 'loss_g'])
            env.update_figure(sample_low_im, x_recon_sample, sample_high_im)
