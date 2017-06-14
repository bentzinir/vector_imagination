import numpy as np
import time
import sys
import matplotlib.pyplot as plt


class MovingBox(object):

    def __init__(self, radius, im_size):

        self.r = radius

        self.size = im_size

        self.vector_dim = 2

        self.iters_vec = []

        self.fig, axarr = plt.subplots(2)

        self.x_obj = axarr[0].imshow(np.random.rand(self.size, self.size))
        self.x_recon_obj = axarr[1].imshow(np.random.rand(self.size, self.size))
        plt.show(block=False)

    def update_figure(self, x, x_recon):
        self.x_obj.set_data(x)
        self.x_obj.set_clim(vmin=x.min(), vmax=x.max())
        self.x_recon_obj.set_data(x_recon)
        self.x_recon_obj.set_clim(vmin=x_recon.min(), vmax=x_recon.max())
        plt.draw()

    def print_info_line(self, itr, attr_list):
        np.set_printoptions(precision=3)
        buf_line = '%s Iter %d' % (time.strftime("%H:%M:%S"), itr)
        for attr in attr_list:
            if type(getattr(self, attr)) == np.ndarray:
                str_format = ', %s: %s'
            else:
                str_format = ', %s: %f'
            buf_line += str_format % (attr, getattr(self, attr))
        buf_line = buf_line.replace("\n", "")
        sys.stdout.write(buf_line + '\n')

    def update_stats(self, abs_grad_norm, abs_weight_norm):
        def update_attr(obj, attr, value):
            if not hasattr(obj, attr):
                setattr(obj, attr, 0.)
            updated_val = self.averaging_factor * getattr(obj, attr) + (1-self.averaging_factor) * value
            setattr(obj, attr, updated_val)

        update_attr(self, 'abs_grad_norm', abs_grad_norm)
        update_attr(self, 'abs_weight_norm', abs_weight_norm)

    def render_image(self, pos, r):
        im = np.zeros((self.size, self.size))
        for i in xrange(self.size):
            for j in xrange(self.size):
                if (([i, j] - pos) ** 2).sum() <= r:
                    im[i, j] = 1
        return im

    def create_batch(self, batch_size):
        batch = []
        positions = []
        for i in xrange(batch_size):
            pos = np.random.randint(low=0, high=self.size, size=2)
            im = self.render_image(pos, self.r)
            batch.append(im.flatten())
            positions.append(pos)
        return np.asarray(batch), positions