import tensorflow as tf
from nn import NN
from moving_box import MovingBox

batch_size = 32
lr = 0.001
layer_widths = [750, 450, 350]
saved_model = None
train_mode = True
n_train_iters = 100000
diaplay_interval = 10
state_coord = 1

env = MovingBox(im_size=12, radius=4)

conv_ae = NN(batch_size=batch_size, x_dim=env.vector_dim, lr=lr)

sess = tf.Session()

init_graph = tf.global_variables_initializer()

sess.run(init_graph)

loss, image, apply_grads = conv_ae.train()

saver = tf.train.Saver()
init_graph = tf.global_variables_initializer()
sess = tf.Session()

if saved_model is None:
    sess.run(init_graph)
else:
    saver.restore(sess, saved_model)

for i in xrange(n_train_iters):
    if train_mode:
        x, positions = env.create_batch(batch_size)
        run_vals = sess.run(fetches=[apply_grads, loss, image],
                            feed_dict={conv_ae.x: positions})

        if i % diaplay_interval == 0:
            x_sample = x[0].reshape((env.size, env.size))
            x_recon_sample = run_vals[2][0].reshape((64, 64))
            # env.print_info_line(i, ['loss'])
            print ('Iter: %d, loss: %f') % (i, run_vals[1])
            env.update_figure(x_sample, x_recon_sample)