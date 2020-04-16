import os
import sys
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import importlib
import tensorflow as tf
from scipy.misc import imsave, imresize
sys.path.append('./facenet')


class Plot_Adversarial_Example:
    def __init__(self, DIR, n_img_x=10, img_w=28, img_h=28, img_c=1, resize_factor=1.0, interval=1):
        self.DIR = DIR
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        assert isinstance(interval, int)
        assert interval >= 1
        self.interval = interval
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
        imsave(os.path.join(self.DIR, name), self._merge(self.img_list[::self.interval]))

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


def output(value_dict, stream=None, bit=3):
    output_str = ''
    for key, value in value_dict.items():
        if isinstance(value, float) or isinstance(value, np.float32) or isinstance(value, np.float64): value = round(value, bit)
        output_str += '[ ' + str(key) + ' ' + str(value) + ' ] '
    print(output_str)
    if stream is not None: print(output_str, file=stream)


def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars): sess.run(tf.variables_initializer(not_initialized_vars))


def main():
    tflib.init_tf()
    IMG_SIZE = 1024
    _G, _D, Gs = pickle.load(open("karras2019stylegan-ffhq-1024x1024.pkl", "rb"))

    rnd = np.random.RandomState(50)
    latents = tf.placeholder(tf.float32, shape=[1, Gs.input_shape[1]], name="init_z")
    adv_z = tf.get_variable('adv_z', shape=(1, Gs.input_shape[1]), dtype=tf.float32)
    images = tf.get_variable('image',shape=(1, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)

    images_adv = tflib.convert_images_to_01(Gs.get_output_for(adv_z, None), nchw_to_nhwc=True)
    module = importlib.import_module('face_recognition')
    facenet = module.Facenet(512)

    facenet_loss, facenet_variables = facenet.build(images, images_adv)
    disz_loss = tf.reduce_mean(tf.maximum(0.35 ** 2 - tf.square(latents - adv_z), 0.0))
    total_loss = 0.001 * facenet_loss + disz_loss
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(total_loss, var_list=[adv_z])
    facenet_saver = tf.train.Saver(var_list=facenet_variables)
    log_file = open('Type I.txt', 'w')
    
    PAE = Plot_Adversarial_Example('results', n_img_x=5, img_w=IMG_SIZE, img_h=IMG_SIZE, img_c=3)  
    with tf.get_default_session() as sess:
        initialize_uninitialized(sess)
        facenet_saver.restore(sess, './facenet/facenet/model-20180402-114759.ckpt-275')
        for k in range(10):
            init_z = rnd.randn(1, Gs.input_shape[1])
            sess.run(tf.assign(adv_z, latents), feed_dict={latents: init_z})
            ori_img_ = sess.run(images_adv)
            PAE.add_image((ori_img_[0]*255).astype(np.uint8))
            sess.run(tf.assign(images, ori_img_))

            for i in range(400):
                _, adv_z_, face_loss_, disz_loss_, images_adv_= sess.run([optimizer, adv_z, facenet_loss, disz_loss, images_adv], feed_dict={latents: init_z})
                distance_z = np.abs(adv_z_ - init_z)
                if i % 40 == 0: PAE.add_image((images_adv_[0]*255).astype(np.uint8))
                output({'Iter': i+1, 'Dist': np.mean(distance_z), 'F Loss': face_loss_, 'D Loss': disz_loss_}, log_file)
                
            images_adv_ = sess.run(images_adv)
            PAE.add_image((images_adv_[0]*255).astype(np.uint8))
            PAE.save_images(name="s%d_%03d_dist_%.4f.png"%(IMG_SIZE, k, face_loss_))
            PAE.clear()


if __name__ == "__main__":
    main()