import tensorflow as tf
from nn import NN
from moving_box import MovingBox
import params

env = MovingBox()

conv_ae = NN(env.vector_dim, [env.im_size, env.im_size])

sess = tf.Session()

init_graph = tf.global_variables_initializer()

sess.run(init_graph)

# generate discriminator graph first
x, x_im, ref_im = env.read_data(params.fname)

d_losses, apply_grads_d, grad_norm_d = conv_ae.train_discriminator(x, ref_im)

g_losses, apply_grads_g, grad_norm_g, fake_im = conv_ae.train_generator(x)

dec_losses, im_fake, apply_grads_ae, grad_norm_ae = conv_ae.train_decoder(x)

init_graph = tf.global_variables_initializer()
sess = tf.Session()
tf.train.start_queue_runners(sess=sess)
env.saver = tf.train.Saver()

if params.model is None:
    sess.run(init_graph)
else:
    env.load(sess)

for i in range(params.n_train_iters):

    data = sess.run([x_im, fake_im, ref_im, x])

    if i % params.display_interval == 0 or params.mode == 'test':
        env.update_figure(x_im=data[0][0], x_fake=data[1][0], x_expert=data[2][0])

    if params.mode == 'train':
        # train discriminator
        for _ in range(params.n_dis_iters):
            run_vals_d = sess.run([apply_grads_d, grad_norm_d, d_losses])

        # train_generator
        for _ in range(params.n_gen_iters):
            run_vals_g = sess.run([apply_grads_g, grad_norm_g]+g_losses)

        # train decoder
        for _ in range(params.n_dec_iters):
            run_vals_dec = sess.run([apply_grads_ae, grad_norm_ae] + dec_losses)

        if i % params.display_interval == 0:
            env.save(sess, i)
            env.update_stats(i, {'loss_d': run_vals_d[2],
                                 'grad_d': run_vals_d[1],
                                 'loss_g': run_vals_g[2],
                                 'grad_g': run_vals_g[1],
                                 'loss_dec': run_vals_dec[2],
                                 'grad_dec': run_vals_dec[1],
                                 })
