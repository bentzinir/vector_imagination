layer_widths = [750, 450, 350]
n_train_iters = 1500
display_interval = 1000
save_interval = 1000
state_coord = 1
lr = 0.00001
weight_decay_rate = 0.0000001
beta1 = 0.5
gan_weight = .1
order_weight = 0.0001
bs = 10
z_dim = 100
min_after_deque = 1000
num_threads = 6
capacity = min_after_deque + (num_threads + 1) * bs
synthetic = False
fname = 'mbox.tfrecords'
model = None
mode = 'train'
create_data = True
n_examples = 1000
n_dis_iters = 1
n_gen_iters = 2
n_dec_iters = 0

if mode == 'test':
    display_interval = 1