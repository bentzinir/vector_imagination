import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import scipy.misc


class MovingBox(object):

    def __init__(self, radius, low_size, high_size):

        self.r = radius

        self.low_size = low_size

        self.vector_dim = 2

        self.high_size = high_size

        self.averaging_factor = 0.5

        self.fig, axarr = plt.subplots(3)

        self.pos = np.random.uniform(low=-1, high=1, size=self.vector_dim)

        self.x_obj = axarr[0].imshow(np.random.rand(self.low_size, self.low_size), interpolation='none', cmap='gray')
        self.x_recon_obj = axarr[1].imshow(np.random.rand(self.high_size, self.high_size), interpolation='none', cmap='gray')
        self.expert_im_obj = axarr[2].imshow(np.random.rand(self.high_size, self.high_size), interpolation='none', cmap='gray')
        plt.show(block=False)

    def resize_image(self, pos):
        return scipy.misc.imresize(self.render_image(np.asarray(pos), self.r), [self.high_size, self.high_size], interp='nearest') / 255.

    def update_figure(self, x_low, x_fake, x_high):
        self.x_obj.set_data(x_low)
        self.x_obj.set_clim(vmin=0., vmax=1.)
        self.x_recon_obj.set_data(np.squeeze(x_fake))
        self.x_recon_obj.set_clim(vmin=0., vmax=1.)
        self.expert_im_obj.set_data(np.squeeze(x_high))
        self.expert_im_obj.set_clim(vmin=0., vmax=1.)
        plt.draw()

    def print_info_line(self, itr, attr_list):
        np.set_printoptions(precision=3)
        buf_line = '%s Iter %d' % (time.strftime("%H:%M:%S"), itr)
        for attr in attr_list:
            str_format = ', %s: %f'
            buf_line += str_format % (attr, getattr(self, attr))
        buf_line = buf_line.replace("\n", "")
        sys.stdout.write(buf_line + '\n')

    def update_stats(self, attr_dict):
        def update_attr(obj, attr, value):
            if not hasattr(obj, attr):
                setattr(obj, attr, 0.)
            updated_val = self.averaging_factor * getattr(obj, attr) + (1-self.averaging_factor) * value
            setattr(obj, attr, updated_val)
        for key, val in attr_dict.iteritems():
            update_attr(self, key, val)

    def render_image(self, pos, r):
        im = np.zeros((self.low_size, self.low_size))
        pos = 0.5 * (np.asarray(pos)+1)*self.low_size
        for i in xrange(self.low_size):
            for j in xrange(self.low_size):
                if abs(([i, j] - pos)).sum() < r:
                    im[i, j] = 1
        return im

    def create_batch(self, batch_size):
        im_low = []
        positions = []
        im_high = []
        for i in xrange(batch_size):
            pos = np.random.uniform(low=-1, high=1, size=self.vector_dim)
            im = self.render_image(pos[:2], self.r)
            im_low.append(im)
            positions.append(pos)
            pos_expert = np.random.uniform(low=-1, high=1, size=2)
            im_expert = self.resize_image(pos_expert)
            im_high.append(np.expand_dims(im_expert, -1))
        return im_low, positions, im_high

