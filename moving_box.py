import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import params


class MovingBox(object):

    def __init__(self, radius=6, im_size=48):

        self.r = radius

        self.im_size = im_size

        self.vector_dim = 2+1

        self.render_image = self.render_gauss_image
        # self.render_image = self.render_square_image

        self.averaging_factor = 0.5

        self.fig, axarr = plt.subplots(3)

        self.x_im_obj = axarr[0].imshow(np.random.rand(self.im_size, self.im_size), interpolation='none', cmap='gray')
        self.x_im_fake_obj = axarr[1].imshow(np.random.rand(self.im_size, self.im_size), interpolation='none', cmap='gray')
        self.expert_im_obj = axarr[2].imshow(np.random.rand(self.im_size, self.im_size), interpolation='none', cmap='gray')
        plt.show(block=False)

        if params.create_data:
            self.create_data(params.fname, params.n_examples)

    def update_obj(self, obj, val):
        val = np.squeeze(val)
        obj.set_data(val)
        obj.set_clim(vmin=val.min(), vmax=val.max())

    def update_figure(self, x_im, x_fake, x_expert):
        self.update_obj(self.x_im_obj, x_im)
        self.update_obj(self.x_im_fake_obj, x_fake)
        self.update_obj(self.expert_im_obj, x_expert)
        plt.draw()
        plt.pause(0.01)

    def print_info_line(self, itr, attr_list):
        np.set_printoptions(precision=3)
        buf_line = '%s Iter %d' % (time.strftime("%H:%M:%S"), itr)
        for attr in attr_list:
            str_format = ', %s: %f'
            buf_line += str_format % (attr, getattr(self, attr))
        buf_line = buf_line.replace("\n", "")
        sys.stdout.write(buf_line + '\n')

    def update_stats(self, itr, attr_dict):
        def update_attr(obj, attr, value):
            if not hasattr(obj, attr):
                setattr(obj, attr, 0.)
            updated_val = self.averaging_factor * getattr(obj, attr) + (1-self.averaging_factor) * value
            setattr(obj, attr, updated_val)
        for key, val in attr_dict.items():
            update_attr(self, key, val)
        self.print_info_line(itr, attr_dict.keys())

    def render_square_image(self, pos, r):
        im = np.zeros((self.im_size, self.im_size))
        pos *= self.im_size
        for i in range(self.im_size):
            for j in range(self.im_size):
                if abs(i - pos[0]) <= r/2. and abs(j - pos[1]) <= r/2.:
                    im[i, j] = 1
        return im

    def render_gauss_image(self, pos, r):
        im = np.zeros((self.im_size, self.im_size))
        pos *= self.im_size
        for i in range(self.im_size):
            for j in range(self.im_size):
                im[i, j] = np.exp(-(((np.asarray([i,j]) - pos)**2).sum())/r**2)
        return im

    def create_example(self):
        pos = np.random.uniform(low=0, high=1, size=2)
        r = np.random.uniform(low=3, high=10, size=1)
        x = np.concatenate([pos, r], axis=0)
        x_im = np.expand_dims(self.render_image(pos, r), 2)
        pos_expert = np.random.uniform(low=0, high=1, size=2)
        reference_im = np.expand_dims(self.render_image(pos_expert, self.r), 2)
        return x, x_im, reference_im

    def create_data(self, fname, n_examples):
        writer = tf.python_io.TFRecordWriter(fname)
        for _ in tqdm(range(n_examples)):
            x, x_im, ref_im = self.create_example()
            # create protobuf object
            example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'x': tf.train.Feature(float_list=tf.train.FloatList(value=x.tolist())),
                        'ref_im': tf.train.Feature(float_list=tf.train.FloatList(value=ref_im.flatten().tolist())),
                        'x_im': tf.train.Feature(float_list=tf.train.FloatList(value=x_im.flatten().tolist())),
                    }
            ))
            # serialize protobuf object before writing to disk
            serialized = example.SerializeToString()
            writer.write(serialized)

    def read_data(self, filename):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                   features={'x': tf.FixedLenFeature([3], tf.float32),
                                             'x_im': tf.FixedLenFeature([self.im_size, self.im_size, 1], tf.float32),
                                             'ref_im': tf.FixedLenFeature([self.im_size, self.im_size, 1], tf.float32),
                                            })
        x = features['x']
        x_im = features['x_im']
        ref_im = features['ref_im']

        x_batch, x_im_batch, ref_im_batch = tf.train.shuffle_batch(
            [x, x_im, ref_im], batch_size=params.bs, capacity=params.capacity, min_after_dequeue=params.min_after_deque)
        return x_batch, x_im_batch, ref_im_batch
