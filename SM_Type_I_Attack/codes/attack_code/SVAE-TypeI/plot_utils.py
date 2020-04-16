import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy.misc import imresize
import os
from moviepy.editor import VideoClip

class Plot_Reproduce_Performance():
    def __init__(self, DIR, n_img_x=8, n_img_y=8, img_w=28, img_h=28, resize_factor=1.0):
        self.DIR = DIR

        assert n_img_x > 0 and n_img_y > 0

        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y

        assert img_w > 0 and img_h > 0

        self.img_w = img_w
        self.img_h = img_h

        assert resize_factor > 0

        self.resize_factor = resize_factor

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_img_x*self.n_img_y, self.img_h, self.img_w)
        imsave(self.DIR + "/"+name, self._merge(images, [self.n_img_y, self.n_img_x]))

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)

        img = np.zeros((h_ * size[0], w_ * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = imresize(image, size=(w_,h_), interp='bicubic')

            img[j*h_:j*h_+h_, i*w_:i*w_+w_] = image_

        return img

class Plot_Manifold_Learning_Result():
    def __init__(self, DIR, n_img_x=20, n_img_y=20, img_w=28, img_h=28, resize_factor=1.0, z_range=4):
        self.DIR = DIR

        assert n_img_x > 0 and n_img_y > 0

        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y

        assert img_w > 0 and img_h > 0

        self.img_w = img_w
        self.img_h = img_h

        assert resize_factor > 0

        self.resize_factor = resize_factor

        assert z_range > 0
        self.z_range = z_range

        self._set_latent_vectors()

    def _set_latent_vectors(self):

        # z1 = np.linspace(-self.z_range, self.z_range, self.n_img_y)
        # z2 = np.linspace(-self.z_range, self.z_range, self.n_img_x)
        #
        # z = np.array(np.meshgrid(z1, z2))
        # z = z.reshape([-1, 2])

        # borrowed from https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py
        z = np.rollaxis(np.mgrid[self.z_range:-self.z_range:self.n_img_y * 1j, self.z_range:-self.z_range:self.n_img_x * 1j], 0, 3)
        # z1 = np.rollaxis(np.mgrid[1:-1:self.n_img_y * 1j, 1:-1:self.n_img_x * 1j], 0, 3)
        # z = z1**2
        # z[z1<0] *= -1
        #
        # z = z*self.z_range

        self.z = z.reshape([-1, 2])

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_img_x*self.n_img_y, self.img_h, self.img_w)
        imsave(self.DIR + "/"+name, self._merge(images, [self.n_img_y, self.n_img_x]))

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)

        img = np.zeros((h_ * size[0], w_ * size[1]))

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = imresize(image, size=(w_, h_), interp='bicubic')

            img[j * h_:j * h_ + h_, i * w_:i * w_ + w_] = image_

        return img

    # borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
    def save_scattered_image(self, z, id, name='scattered_image.jpg'):
        N = 10
        plt.figure(figsize=(8, 6))
        plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
        plt.colorbar(ticks=range(N))
        axes = plt.gca()
        axes.set_xlim([-self.z_range-2, self.z_range+2])
        axes.set_ylim([-self.z_range-2, self.z_range+2])
        plt.grid(True)
        plt.savefig(self.DIR + "/" + name)


class Plot_Adversarial_Example:
    def __init__(self, DIR, n_img_x=10, img_w=28, img_h=28, img_c=1, resize_factor=1.0):
        self.DIR = DIR
        if not os.path.exists(DIR):
            os.makedirs(DIR)

        assert n_img_x > 0

        self.n_img_x = n_img_x

        assert img_w > 0 and img_h > 0

        self.img_w = img_w
        self.img_h = img_h

        assert resize_factor > 0

        self.resize_factor = resize_factor

        assert img_c == 1 or img_c == 3

        self.img_c = img_c

        self.img_list = []

    def save_images(self, name='result.jpg'):
        imsave(os.path.join(self.DIR, name), self._merge(self.img_list))

    def _merge(self, image_list):

        size_y = len(image_list) // self.n_img_x + (1 if len(image_list) % self.n_img_x != 0 else 0)
        size_x = self.n_img_x

        h_ = int(self.img_h * self.resize_factor)
        w_ = int(self.img_w * self.resize_factor)

        img = np.zeros((h_ * size_y, w_ * size_x, self.img_c))

        for idx, image in enumerate(image_list):
            i = int(idx % size_x)
            j = int(idx / size_x)

            image_ = imresize(image, size=(w_,h_), interp='bicubic')
            img[j*h_:j*h_+h_, i*w_:i*w_+w_, :] = image_.reshape((self.img_w, self.img_h, self.img_c))

        return img.squeeze()

    def add_image(self, img):
        self.img_list.append(img)

    def add_images(self, img_list):
        self.img_list = img_list

    def clear(self):
        self.img_list = []


class Image_Animation(object):
    def __init__(self, DIR, data, fps=20):
        self.DIR = DIR
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        self.data = data
        self.fps = fps

        self._index = 0
        self.num_data = len(data)

    def make_frame(self, t):
        frame = self.data[self._index]
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, 2)
            frame = np.repeat(frame, 3, 2)

        frame = imresize(frame, size=(480, 480), interp='bicubic')
        self._index += 1
        return frame

    def save_animation(self, name):
        self._index = 0
        duration = self.num_data // self.fps
        anim = VideoClip(make_frame=self.make_frame, duration=duration)
        # anim.write_gif(os.path.join(self.DIR, name), fps=self.fps)
        anim.write_videofile(os.path.join(self.DIR, name), fps=self.fps, audio=False)
        print(duration)



# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)