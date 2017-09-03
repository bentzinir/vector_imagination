checkpoints_dir = 'checkpoints/'
layer_widths = [750, 450, 350]
n_train_iters = 70000
display_interval = 1000
save_interval = 1000
state_coord = 1
lr = 0.00001
weight_decay_rate = 0.0000001
beta1 = 0.5
gan_weight = .1
tv_weight = 0.0000
dec_weight = 0.05
order_weight = 0.0001
bs = 10
z_dim = 100
synthetic = False
fname = 'mbox.tfrecords'
model = None
mode = 'train'
create_data = False
n_examples = 10000
n_dis_iters = 1
n_gen_iters = 2
n_dec_iters = 1

if mode == 'test':
    display_interval = 1