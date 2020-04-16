import tensorflow as tf
import numpy as np
import importlib
import os
from scipy import misc
import scipy.misc
from PIL import Image

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def getimg(path):
    files = os.listdir(path)
    assert len(files) == 2, 'image num !=2'
    files.sort()
    batch = len(files)
    data = np.zeros((batch, 160, 160, 3), np.float32)
    print('load image')
    for i in range(batch):
        imgpath = os.path.join(path, files[i])
        print(imgpath)
        img = misc.imread(imgpath, mode='RGB')
        data[i] = img
    return data

def getimg_org(path):
    img = misc.imread(path, mode='RGB')
    #data = np.zeros((2, img.shape[0], int(img.shape[1] / 2), 3), np.float32)
    img_org = img[:, :int(img.shape[1] / 2), :]

    return img_org

def getimg_tsl(path):
    img = misc.imread(path, mode='RGB')
    data = np.zeros((2, img.shape[0], int(img.shape[1] / 2), 3), np.float32)
    img_org = img[:, :int(img.shape[1] / 2), :]
    img_attack = img[:, int(img.shape[1] / 2):, :]
    data[0] = img_org
    data[1] = img_attack
    misc.imsave('org.jpg', img_org)
    misc.imsave('attack.jpg', img_attack)
    return data


def getimg_one(path):
    img = misc.imread(path, mode='RGB')

    return img


class Facenet:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size

    def build(self, x1, x2):
        x1 = tf.image.resize_bilinear(x1, [160, 160])

        x2 = tf.image.resize_bilinear(x2, [160, 160])


        x1 = self.prewhiten(x1)
        x2 = self.prewhiten(x2)

        image_batch = tf.concat([x1, x2], axis=0)
        network = importlib.import_module('models.inception_resnet_v1')
        prelogits, _ = network.inference(image_batch, keep_probability=1, phase_train=False,
                                         bottleneck_layer_size=self.embedding_size)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        variables = dict()
        variables['facenet'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionResnetV1')

        loss = dict()

        loss['dist'] = tf.sqrt(tf.reduce_sum(tf.square(embeddings[0, :] - embeddings[1, :])))

        return loss, None, None, variables

    def build_one(self, x1):
        x1 = tf.image.resize_bilinear(x1, [160, 160])

        x1 = self.prewhiten(x1)

        image_batch = x1
        network = importlib.import_module('models.inception_resnet_v1')
        prelogits, _ = network.inference(image_batch, keep_probability=1, phase_train=False,
                                         bottleneck_layer_size=self.embedding_size)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        variables = dict()
        variables['facenet'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionResnetV1')

        return embeddings, variables

    def build_one_reuse(self, x1):
        x1 = tf.image.resize_bilinear(x1, [160, 160])

        x1 = self.prewhiten(x1)

        image_batch = x1
        network = importlib.import_module('models.inception_resnet_v1')
        prelogits, _ = network.inference(image_batch, keep_probability=1, phase_train=False,
                                         bottleneck_layer_size=self.embedding_size, reuse=True)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        variables = dict()
        variables['facenet'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionResnetV1')

        return embeddings, variables

    def prewhiten(self, x):
        mean, var = tf.nn.moments(x, axes=[0, 1, 2, 3])
        std = tf.sqrt(var)
        size = tf.cast(x.shape[1] * x.shape[2] * x.shape[3], tf.float32)

        std_adj = tf.maximum(std, 1.0 / tf.sqrt(size))
        y = tf.multiply(tf.subtract(x, mean), 1 / std_adj)
        return y



def generate_pgd(x):
    perturbation_multiplier = 1.0
    # bounds=(-1,1)
    bounds = (0, 255)

    epsilon = 20#5  # 0.1
    step_size = 2.55#2.55  # 10
    niter = 10#10  # 2

    # rescale epsilon and step size to image bounds
    epsilon = float(epsilon) / 255.0 * (bounds[1] - bounds[0])
    step_size = float(step_size) / 255.0 * (bounds[1] - bounds[0])

    # clipping boundaries
    clip_min = tf.maximum(x - epsilon, bounds[0])
    clip_max = tf.minimum(x + epsilon, bounds[1])

    # compute starting point
    start_x = x + tf.random_uniform(tf.shape(x), -epsilon, epsilon)
    start_x = tf.clip_by_value(start_x, clip_min, clip_max)

    # main iteration of PGD
    loop_vars = [0, start_x]

    y1, variables = face.build_one_reuse(x)
    def loop_cond(index, _):
        return index < niter


    def loop_body(index, adv_images):
        #print(index)
        logits, variables = face.build_one_reuse(adv_images)
        loss = tf.sqrt(tf.reduce_sum(tf.square(y1 - logits)))
        """
        loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=y1,
                logits=logits))
        """
        print(loss,adv_images)
        perturbation = step_size * tf.sign(tf.gradients(loss, adv_images)[0])
        new_adv_images = adv_images + perturbation_multiplier * perturbation
        new_adv_images = tf.clip_by_value(new_adv_images, clip_min, clip_max)
        return index + 1, new_adv_images


    with tf.control_dependencies([start_x]):
        _, result = tf.while_loop(
            loop_cond,
            loop_body,
            loop_vars,
            back_prop=False,
            parallel_iterations=1)
        return result,variables


if __name__ == '__main__':

    face = Facenet(512)
    image1 = tf.get_variable(name='image1', shape=(1, 64, 64, 3), dtype=tf.float32)
    image1_typeI = tf.get_variable(name='image1_1', shape=(1, 64, 64, 3), dtype=tf.float32)


    img1_pl = tf.placeholder(dtype=tf.float32, shape=(1, 64, 64, 3), name='img1_pl')
    img1_pl_typeI = tf.placeholder(dtype=tf.float32, shape=(1, 64, 64, 3), name='img1_pl_1')

    assign_op1 = tf.assign(image1, img1_pl)
    assign_op1_typeI = tf.assign(image1_typeI, img1_pl_typeI)

    y1_typeI, variables = face.build_one(image1_typeI)
    y1,variables=face.build_one_reuse(image1)

    dist_ori = tf.sqrt(tf.reduce_sum(tf.square(y1 - y1_typeI)))

    with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_list=variables['facenet'])

            saver.restore(sess, "D:/research/facenet-master/src/20190329-111621/model-20190329-111621.ckpt-1")


            path = "D:/research/facenet_cmj/face/typeII_mpgd/"#./face/same/ D:/research/facenet-master/src/data/100/Antony_Leung
            pathout = "./face/typeI_new/"
            if not os.path.exists(pathout):
                os.makedirs(pathout)

            flag = 0
            total = 0
            right = 0
            num=0
            for filename in os.listdir(path):

              if num>489:
                img_path = os.path.join(path, filename)
                data = getimg_tsl(img_path)
                img1=data[0].reshape([1, 64, 64, 3])
                img2=data[1].reshape([1, 64, 64, 3])

                #print(img1.shape)
                #img1 = tf.random_crop(img1, [160, 160, 3])


                sess.run([assign_op1,assign_op1_typeI], feed_dict={img1_pl: img1, img1_pl_typeI: img2})


                finalimage,dist= sess.run([image1_typeI,dist_ori])
                orgimage=sess.run(image1)
                if dist<1.1:
                    right+=1
                total+=1

                finalimage = np.squeeze(finalimage)
                orgimage=np.squeeze(orgimage)


                orgfinal =np.concatenate((orgimage,finalimage),axis=1)


                for i in range(100):
                    if filename[i] == ']':
                        break
                print(filename[1:i])
                #num = int(filename[1:i])
                print(os.path.join(pathout, filename[1:i]))

                #scipy.misc.imsave(os.path.join(pathout, '[{}]_distorg_[{:.6}].bmp'.format(filename[1:i], dist)),
                                 # orgfinal)

                flag += 1

              num+=1
            print(right, total)
            print(right / total)


