import tensorflow as tf
import numpy as np
import celebA_data
import plot_utils
import os
import celebA_model
import importlib
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '../../facenet')
from face_recognition import Facenet

IMAGE_SIZE = 64
np.random.seed(996)
tf.set_random_seed(996)


def get_string(label):
    if label == 0:
        return 'female'
    else:
        return 'male'


def train(args):
    classes = args.classes
    dim_z = args.dim_z
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    data_dir = args.data_dir

    RESULT_DIR = args.results_path
    DATASET = 'celebA-dim_z{}'.format(dim_z)
    TEMP_DIR = './tmp'

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(os.path.join(RESULT_DIR, DATASET)):
        os.makedirs(os.path.join(RESULT_DIR, DATASET))

    print('loading celebA dataset ...')
    x_train, y_train, x_test, y_test = celebA_data.load_celebA_Gender(data_dir=data_dir)
    print('done!')

    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input_img')
    y = tf.placeholder(tf.float32, shape=[None, 1], name='input_label')

    machine = celebA_model.CelebA_Model(dim_z, classes)
    loss, optim, metric, variables = machine.build(x, y, batch_size, learning_rate)

    n_train_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    total_train_batch = int(n_train_samples / batch_size)
    total_test_batch = int(n_test_samples / batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if True:  # stage 1: train the supervised variational autoencoder
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

            # restore training
            # saver.restore(sess, os.path.join(RESULT_DIR, DATASET, "{}_epoch_170.data".format(DATASET)))

            for epoch in range(0, epochs+1):
                # Random shuffling
                indexes = np.arange(0, n_train_samples)
                np.random.shuffle(indexes)
                x_train = x_train[indexes, ...]
                y_train = y_train[indexes, ...]
                for i in range(total_train_batch):
                    # Compute the offset of the current minibatch in the data.
                    offset = (i * batch_size) % n_train_samples
                    batch_xs = x_train[offset:(offset + batch_size), ...]
                    batch_ys = y_train[offset:(offset + batch_size), ...]

                    feed_dict = {x: batch_xs, y: batch_ys}
                    # update sae
                    _, = sess.run([optim['SAE']], feed_dict=feed_dict)

                    if i % 100 == 0:
                        loss_tot, d_cla, d_KL, g_rec, acc = sess.run(
                            [loss['SAE'], loss['d_cla'],  loss['d_KL'], loss['g_rec'], metric['acc']],
                            feed_dict=feed_dict)
                        msg = '[stage1] epoch:{}/{} step:{}/{} '.format(epoch, epochs, i, total_train_batch)
                        msg += 'loss_tot:{:.4}, d_cla:{:.4}, d_KL:{:.4}, g_rec:{:.4}, acc:{:.4}'.format(loss_tot, d_cla, d_KL, g_rec, acc)
                        print(msg)

                # validate after each epoch
                val_loss_tot, val_d_cla, val_d_KL, val_g_rec, val_acc = 0.0, 0.0, 0.0, 0.0, 0.0

                for i in range(total_test_batch):
                    offset = (i * batch_size) % n_test_samples
                    batch_xs = x_test[offset:(offset + batch_size), ...]
                    batch_ys = y_test[offset:(offset + batch_size), ...]
                    feed_dict = {x: batch_xs, y: batch_ys}
                    loss_tot, d_cla, d_KL, g_rec, acc = sess.run(
                        [loss['SAE'], loss['d_cla'], loss['d_KL'], loss['g_rec'], metric['acc']],
                        feed_dict=feed_dict)

                    val_loss_tot += loss_tot
                    val_d_cla += d_cla
                    val_d_KL += d_KL
                    val_g_rec += g_rec
                    val_acc += acc

                msg = '[stage1] epoch:{}/{} '.format(epoch, epochs)
                msg += 'val_loss_tot:{:.4}, val_d_cla:{:.4}, val_d_KL:{:.4}, val_g_rec:{:.4}'.format(val_loss_tot / total_test_batch,
                                                                                                     val_d_cla / total_test_batch,
                                                                                                     val_d_KL / total_test_batch,
                                                                                                     val_g_rec / total_test_batch)
                msg += 'val_acc:{:.4}'.format(val_acc / total_test_batch)
                print(msg)

                batch_xs = x_test[0:100, ...]
                x_rec, x_gen = sess.run([metric['x_hat'], metric['x_fake_hat']], feed_dict={x: batch_xs})
                PAE = plot_utils.Plot_Adversarial_Example(os.path.join(TEMP_DIR, DATASET), img_w=IMAGE_SIZE, img_h=IMAGE_SIZE, img_c=3)
                PAE.add_images(list(celebA_data.deprocess(batch_xs)))
                PAE.save_images(name="{}_ori.png".format(str(epoch).zfill(3)))
                PAE.clear()
                PAE.add_images(list(celebA_data.deprocess(x_rec)))
                PAE.save_images(name="{}_rec.png".format(str(epoch).zfill(3)))
                PAE.clear()
                PAE.add_images(list(celebA_data.deprocess(x_gen)))
                PAE.save_images(name="{}_gen.png".format(str(epoch).zfill(3)))

                if epoch % 10 == 0:
                    saver.save(sess, os.path.join(RESULT_DIR, DATASET, "{}_epoch_{}.data".format(DATASET, epoch)))

            saver.save(sess, os.path.join(RESULT_DIR, DATASET, "{}_stage1.data".format(DATASET)))

        if True:  # stage2: train a discriminator to estimate the manifold in the gaussian latent space
            saver_loader = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=variables['enc']+variables['gen']+variables['cla'])
            saver_loader.restore(sess, os.path.join(RESULT_DIR, DATASET, '{}_stage1.data'.format(DATASET)))
            saver_writer = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=variables['dis'])

            for epoch in range(0, epochs+1):
                # Random shuffling
                indexes = np.arange(0, n_train_samples)
                np.random.shuffle(indexes)
                x_train = x_train[indexes, ...]
                y_train = y_train[indexes, ...]
                for i in range(total_train_batch):
                    # Compute the offset of the current minibatch in the data.
                    offset = (i * batch_size) % n_train_samples
                    batch_xs = x_train[offset:(offset + batch_size), ...]
                    batch_ys = y_train[offset:(offset + batch_size), ...]

                    feed_dict = {x: batch_xs, y: batch_ys}
                    # update sae
                    _ = sess.run(optim['DIS'], feed_dict=feed_dict)

                    if i % 100 == 0:
                        loss_dis, acc_dis_true, acc_dis_fake = sess.run(
                            [loss['dis'], metric['acc_dis_true'],  metric['acc_dis_fake']],
                            feed_dict=feed_dict)
                        msg = '[stage2] epoch:{}/{} step:{}/{} '.format(epoch, epochs, i, total_train_batch)
                        msg += 'loss_dis:{:.4}, acc_dis_true:{:.4}, acc_dis_fake:{:.4}'.format(loss_dis, acc_dis_true, acc_dis_fake)
                        print(msg)

                # validate after each epoch
                val_loss_dis, acc_dis_true, acc_dis_fake = 0.0, 0.0, 0.0

                for i in range(total_test_batch):
                    offset = (i * batch_size) % n_train_samples
                    batch_xs = x_test[offset:(offset + batch_size), ...]
                    batch_ys = y_test[offset:(offset + batch_size), ...]
                    feed_dict = {x: batch_xs, y: batch_ys}

                    loss_dis, acc_dis_true, acc_dis_fake = sess.run(
                        [loss['dis'], metric['acc_dis_true'], metric['acc_dis_fake']],
                        feed_dict=feed_dict)

                    val_loss_dis += loss_dis
                    acc_dis_true += acc_dis_true
                    acc_dis_fake += acc_dis_fake

                msg = '[stage2] epoch:{}/{} '.format(epoch, epochs)
                msg += 'val_loss_dis:{:.4}, acc_dis_true:{:.4}, acc_dis_fake:{:.4}'.format(val_loss_dis / total_test_batch,
                                                                                           acc_dis_true / total_test_batch,
                                                                                           acc_dis_fake / total_test_batch)
                print(msg)
                if epoch % 10 == 0:
                    saver_writer.save(sess, os.path.join(RESULT_DIR, DATASET, "{}_stage2_epoch_{}.data".format(DATASET, epoch)))
            saver_writer.save(sess, os.path.join(RESULT_DIR, DATASET, "{}_stage2.data".format(DATASET)))


def transition(args):
    test_img_index = args.test_index  # choose test image
    verbose_anchor = [0, 500, 3000, 8000, 10000]

    classes = args.classes
    dim_z = args.dim_z
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    RESULT_DIR = args.results_path
    DATASET = 'celebA-dim_z{}'.format(dim_z)
    TEMP_DIR = './tmp'
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(os.path.join(RESULT_DIR, DATASET)):
        os.makedirs(os.path.join(RESULT_DIR, DATASET))

    # load dataset
    _, _, x_test, y_test = celebA_data.load_celebA_Gender(data_dir=args.data_dir, test_only=True)

    # choose test image and its target label

    test_img = x_test[test_img_index, ...]
    test_label = (y_test[test_img_index, ...]).squeeze()
    print('original_label is :', get_string(test_label))
    target_img_label = 1 - test_label
    target_label = np.zeros(shape=[1, 1])
    target_label[0, 0] = target_img_label
    print("target_label is ", get_string(target_img_label))
    # build the testing network
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input_img')
    y = tf.placeholder(tf.float32, shape=[None, 1], name='input_label')

    z = tf.get_variable('z_var', shape=[1, dim_z], dtype=tf.float32)

    machine = celebA_model.CelebA_Model(dim_z, classes)
    loss, optim, metric, variables = machine.build(x, y, batch_size, learning_rate)

    logits = machine.classify(z)
    x_hat = machine.decode(z)

    dis_logits = machine.discriminate(z)
    l_dis = tf.squeeze(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_logits, labels=tf.ones_like(dis_logits)))

    l_cla = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
    l_tot = l_cla + 0.01 * l_dis + tf.norm(z) * 0.0005

    opt = tf.train.AdamOptimizer(5e-3).minimize(l_tot, var_list=[z])

    assign_op = tf.assign(z, metric['latent'])

    with tf.Session() as sess:

        svae_saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=variables['enc'] + variables['gen'] + variables['cla'])
        mc_gan_saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=variables['dis'])
        sess.run(tf.global_variables_initializer())

        # restore stage1 oracle classifier
        svae_saver.restore(sess, os.path.join(RESULT_DIR, DATASET, '{}_stage1.data'.format(DATASET)))

        # restore stage2 monte carlo discriminator
        mc_gan_saver.restore(sess, os.path.join(RESULT_DIR, DATASET, '{}_stage2.data'.format(DATASET)))

        # initialize z_var
        _ = sess.run(assign_op, feed_dict={x: np.expand_dims(test_img, 0)})

        # only for interpolation
        val_z = sess.run(z)
        np.save('[{}]z_0.npy'.format(test_img_index), val_z)

        # plot utils
        PAE = plot_utils.Plot_Adversarial_Example(os.path.join(RESULT_DIR, 'images/'), n_img_x=len(verbose_anchor)+1, img_w=IMAGE_SIZE, img_h=IMAGE_SIZE, img_c=3)
        PAE.add_image(celebA_data.deprocess(test_img).squeeze())
        loss_cla_buf = []
        dis_buf = []
        verbose_period = 100
        iterations = 10000
        for i in range(iterations+1):
            if i in verbose_anchor: #  i % verbose_period == 0:
                x_rec, loss_cla, dis = sess.run([x_hat, l_cla, l_dis], feed_dict={y: target_label})
                PAE.add_image(celebA_data.deprocess(x_rec).squeeze())
                loss_cla_buf.append(np.log(loss_cla + 1e-5))
                dis_buf.append(np.log(dis + 1e-5))

            _, loss_tot, loss_cla, val_dis, val_z = sess.run([opt, l_tot, l_cla, l_dis, z], feed_dict={y: target_label})
            if i % verbose_period == 0:
                msg = 'step {}, loss: {:.6}, loss_cla:{:.6}, val_dis: {:.6}, max_z: {:.4}' \
                    .format(i, loss_tot, loss_cla, val_dis, np.max(np.abs(val_z)))
                print(msg)

        val_z = sess.run(z)
        np.save('[{}]z_{}.npy'.format(test_img_index, iterations), val_z)

        # IA = plot_utils.Image_Animation(os.path.join(RESULT_DIR, 'images/'), PAE.img_list)
        # IA.save_animation('{}_to_{}.mp4'.format(test_label, target_img_label))
        PAE.save_images('[{}]{}_to_{}_rec.jpg'.format(test_img_index, test_label, target_img_label))

        plt.plot(np.arange(0, verbose_period * len(loss_cla_buf), verbose_period), loss_cla_buf, c='r', label='oracle classifier loss')
        plt.plot(np.arange(0, verbose_period * len(dis_buf), verbose_period), dis_buf, c='b', label='discriminator loss')
        plt.xlabel('steps')
        plt.ylabel('log loss')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(RESULT_DIR, 'images/', '[{}]loss_{}_to_{}.jpg'.format(test_img_index, test_label, target_img_label)))

        print('transition outputs to {}'.format(os.path.join(RESULT_DIR, 'images/')))


def interpolation(args):  # test interpolation in VAE
    verbose_anchor = [0, 500, 3000, 8000, 10000]
    test_img_index = 110
    val_z_0 = np.load('[{}]z_0.npy'.format(test_img_index))
    val_z_20k = np.load('[{}]z_{}.npy'.format(test_img_index, verbose_anchor[-1]))

    classes = args.classes
    dim_z = args.dim_z
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    RESULT_DIR = args.results_path
    DATASET = 'celebA-dim_z{}'.format(dim_z)
    TEMP_DIR = './tmp'
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(os.path.join(RESULT_DIR, DATASET)):
        os.makedirs(os.path.join(RESULT_DIR, DATASET))

    _, _, x_test, y_test = celebA_data.load_celebA_Gender(data_dir=args.data_dir, test_only=True)
    test_img = x_test[test_img_index, ...]

    # build the testing network
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='input_img')
    y = tf.placeholder(tf.float32, shape=[None, 1], name='input_label')

    z = tf.get_variable('z_var', shape=[1, dim_z], dtype=tf.float32)

    svae = celebA_model.CelebA_Model(dim_z, classes)
    loss, optim, metric, variables = svae.build(x, y, batch_size, learning_rate)
    x_hat = svae.decode(z)

    with tf.Session() as sess:

        svae_saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=variables['enc'] + variables['gen'] + variables['cla'])
        mc_gan_saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=variables['dis'])
        sess.run(tf.global_variables_initializer())

        # load the parameters of svae
        svae_saver.restore(sess, os.path.join(RESULT_DIR, DATASET, '{}_stage1.data'.format(DATASET)))

        # load the parameters of discrminator on manifold
        mc_gan_saver.restore(sess, os.path.join(RESULT_DIR, DATASET, '{}_stage2.data'.format(DATASET)))

        # plot utils
        PAE = plot_utils.Plot_Adversarial_Example(os.path.join(RESULT_DIR, 'images/'), n_img_x=12)
        PAE.add_image(celebA_data.deprocess(test_img).squeeze())
        iterations = verbose_anchor[-1]
        for i in verbose_anchor:
            z_val = val_z_0 + (val_z_20k - val_z_0) * (i * 1.0 / iterations)
            x_rec = sess.run(x_hat, feed_dict={z: z_val})
            PAE.add_image(celebA_data.deprocess(x_rec).squeeze())

        PAE.save_images('[{}]interpolation.jpg'.format(test_img_index))

        print('interpolation outputs to {}'.format(os.path.join(RESULT_DIR, 'images/')))


def attack_facenet(args):
    test_img_index = args.test_index  # choose test image

    classes = args.classes
    dim_z = args.dim_z
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    RESULT_DIR = args.results_path
    DATASET = 'celebA-dim_z{}'.format(dim_z)
    TEMP_DIR = './tmp'
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(os.path.join(RESULT_DIR, DATASET)):
        os.makedirs(os.path.join(RESULT_DIR, DATASET))

    # load dataset
    _, _, x_test, y_test = celebA_data.load_celebA_Gender(data_dir=args.data_dir, test_only=True)

    # choose test image and its target label

    test_img = x_test[test_img_index, ...]
    test_label = (y_test[test_img_index, ...]).squeeze()
    print('original_label is :', get_string(test_label))
    target_img_label = 1 - test_label
    target_label = np.zeros(shape=[1, 1])
    target_label[0, 0] = target_img_label
    print("target_label is ", get_string(target_img_label))
    # build the testing network
    x = tf.placeholder(tf.float32, shape=[1, IMAGE_SIZE, IMAGE_SIZE, 3], name='input_img')
    y = tf.placeholder(tf.float32, shape=[1, 1], name='input_label')

    z = tf.get_variable('z_var', shape=[1, dim_z], dtype=tf.float32)
    k = tf.placeholder(tf.float32, shape=())
    val_kt = 0
    machine = celebA_model.CelebA_Model(dim_z, classes)
    loss, optim, metric, variables = machine.build(x, y, batch_size, learning_rate)

    logits = machine.classify(z)
    x_hat = machine.decode(z)

    # build the facenet
    facenet = Facenet(512)
    facenet_loss, _, _, facenet_variables = facenet.build(x, x_hat)

    dis_hat = tf.sigmoid(tf.squeeze(machine.discriminate(z)))
    l_dis = - tf.log(dis_hat)

    l_cla = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
    l_tot = l_cla + 0.01 * l_dis + tf.norm(z) * 0.0001 + facenet_loss['dist'] * k
    opt = tf.train.AdamOptimizer(1e-2).minimize(l_tot, var_list=[z])

    assign_op = tf.assign(z, metric['latent'])

    with tf.Session() as sess:

        svae_saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=variables['enc'] + variables['gen'] + variables['cla'])
        mc_gan_saver = tf.train.Saver(write_version=tf.train.SaverDef.V1, var_list=variables['dis'])
        facenet_saver = tf.train.Saver(var_list=facenet_variables['facenet'])
        sess.run(tf.global_variables_initializer())

        # restore stage1 oracle classifier
        svae_saver.restore(sess, os.path.join(RESULT_DIR, DATASET, '{}_stage1.data'.format(DATASET)))

        # restore stage2 monte carlo discriminator
        mc_gan_saver.restore(sess, os.path.join(RESULT_DIR, DATASET, '{}_stage2.data'.format(DATASET)))

        # restore facenet
        facenet_saver.restore(sess, '../../facenet/facenet/20180402-114759/model-20180402-114759.ckpt-275')

        # initialize z_var
        _ = sess.run(assign_op, feed_dict={x: np.expand_dims(test_img, 0)})

        # plot utils
        PAE = plot_utils.Plot_Adversarial_Example(os.path.join(RESULT_DIR, 'images/'), n_img_x=20, img_w=IMAGE_SIZE, img_h=IMAGE_SIZE, img_c=3)
        PAE.add_image(celebA_data.deprocess(test_img).squeeze())
        verbose_period = 100
        for i in range(14999):
            if i % verbose_period == 0:
                x_rec = sess.run(x_hat)
                PAE.add_image(celebA_data.deprocess(x_rec).squeeze())
           
            _, loss_tot, loss_cla, val_dis, val_z, loss_facenet = sess.run([opt, l_tot, l_cla, dis_hat, z, facenet_loss['dist']],
                                                                 feed_dict={y: target_label, x: np.expand_dims(test_img, 0),k: val_kt})
            val_kt += 1e-4*(val_dis*1e-3-loss_cla + np.maximum(val_dis-1.0, 0))
            val_kt = np.clip(val_kt, 0.0, 0.005)
            if i % verbose_period == 0:
                msg = 'step {}, loss: {:.6}, loss_cla:{:.6}, val_dis: {:.6}, max_z: {:.4}, loss_facenet: {:.4}, kt:{:.4}' \
                    .format(i, loss_tot, loss_cla, val_dis, np.max(np.abs(val_z)), loss_facenet, val_kt)
                print(msg)

        # IA = plot_utils.Image_Animation(os.path.join(RESULT_DIR, 'images/'), PAE.img_list)
        # IA.save_animation('[{}]_{}_to_{}_attack.mp4'.format(test_img_index, test_label, target_img_label))
        PAE.save_images('[{}]{}_to_{}_attack.jpg'.format(test_img_index, test_label, target_img_label))

        print('attack outputs to {}'.format(os.path.join(RESULT_DIR, 'images/')))

        PAE = plot_utils.Plot_Adversarial_Example(os.path.join(RESULT_DIR, 'images/'), n_img_x=2, img_w=IMAGE_SIZE, img_h=IMAGE_SIZE, img_c=3)
        x_img, dist = sess.run([x_hat, facenet_loss['dist']], feed_dict={x: np.expand_dims(test_img, 0)})
        x_img_changed = celebA_data.deprocess(x_img).squeeze()
        x_img_original = celebA_data.deprocess(test_img).squeeze()
        PAE.add_image(x_img_original)
        PAE.add_image(x_img_changed)
        PAE.save_images('[{}]{}_to_{}_attack_dist[{:.4}].jpg'.format(test_img_index, test_label, target_img_label, dist))
