"""
Created on Fri Apr 27 22:28:26 2018

@author: cmj
"""
import tensorflow as tf
import numpy as np
import importlib
import os
from scipy import misc

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y 

def getimg(path):
    files = os.listdir(path)
    assert len(files)==2,'image num !=2'
    files.sort()
    batch=len(files)
    data = np.zeros((batch, 160, 160, 3), np.float32) 
    print ('load image')
    for i in range(batch):
        imgpath=os.path.join(path, files[i])        
        print (imgpath)        
        img = misc.imread(imgpath, mode='RGB')
        data[i]=img    
    return data


class Facenet:
    def __init__(self, embedding_size):
        self.embedding_size = embedding_size

    def build(self, x1, x2):
        x1 = tf.image.resize_bilinear(x1, [160,160])
        x2 = tf.image.resize_bilinear(x2, [160,160])

        x1 = self.prewhiten(x1)
        x2 = self.prewhiten(x2)

        image_batch = tf.concat([x1, x2], axis=0)
        network = importlib.import_module('models.inception_resnet_v1')
        prelogits, _ = network.inference(image_batch, keep_probability=1, phase_train=False, bottleneck_layer_size=self.embedding_size)
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        
        variables = dict()
        variables['facenet'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionResnetV1')

        loss = dict()
        loss['dist'] = tf.sqrt(tf.reduce_sum(tf.square(embeddings[0, :] - embeddings[1, :])))

        return loss, None, None, variables

    def prewhiten(self, x):
        mean, var = tf.nn.moments(x, axes=[0, 1, 2, 3])
        std = tf.sqrt(var)
        size = tf.cast(x.shape[1]*x.shape[2]*x.shape[3], tf.float32)
        std_adj = tf.maximum(std, 1.0/tf.sqrt(size))
        y = tf.multiply(tf.subtract(x, mean), 1/std_adj)
        return y 


def face_l2_dist(image,embedding_size=512):
     image[0]=prewhiten(image[0])
     image[1]=prewhiten(image[1])
     with tf.Graph().as_default():
        with tf.Session() as sess:  
            network = importlib.import_module('models.inception_resnet_v1')
            image_batch = tf.placeholder(tf.float32,shape=(None,160,160,3), name='image_batch')
            prelogits, _ = network.inference(image_batch, keep_probability=1, 
                    phase_train=False, bottleneck_layer_size=embedding_size)   
            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        
            sess.run(tf.global_variables_initializer())
            
            saver = tf.train.Saver()
            #saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
            saver.restore(sess, "./facenet/20180402-114759/model-20180402-114759.ckpt-275")
            
            emb=sess.run(embeddings,feed_dict={image_batch:image})

            dist = np.sqrt(np.sum(np.square(np.subtract(emb[0,:], emb[1,:]))))
            
            print (dist)                     
#            nrof_images = image.shape[0]            
            # Print distance matrix
#            print('Distance matrix')
#            print('    ', end='')
#            for i in range(nrof_images):
#                print('    %1d     ' % i, end='')
#            print('')
#            for i in range(nrof_images):
#                print('%1d  ' % i, end='')
#                for j in range(nrof_images):
#                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
#                    print('  %1.4f  ' % dist, end='')
#                print('')
            return dist 



if __name__=='__main__':    
    image=getimg('./LFWTEST/diff4/')
    face_l2_dist(image)
    image=getimg('./LFWTEST/diff4/')
    face=Facenet(512)
    with tf.Graph().as_default():
        with tf.Session() as sess: 
            image= tf.convert_to_tensor(image)
            dist,_,_,_ =face.build(image[0:1],image[1:2])
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            #saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
            saver.restore(sess, "./facenet/20180402-114759/model-20180402-114759.ckpt-275")
            res=sess.run(dist)
    print (res)    