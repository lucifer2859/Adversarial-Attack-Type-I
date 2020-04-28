import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

import mnist_model
import mnist_data
import itertools

np.random.seed(996)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

EPSILON = 1e-6

IMAGE_SIZE = 28

epoch_num = [600, 600]
pretrain_epoch_num = [-1, -1]
stage_flag = [True, True]
batch_size = 64
dim_z = 32
classes = 10
lr = 0.0002
z_lr = 0.005
J1_hat = 0.01

if not os.path.exists('out/'):
    os.makedirs('out/')

if not os.path.exists('out/mnist/'):
    os.makedirs('out/mnist/')

if not os.path.exists('attack/'):
    os.makedirs('attack/')

if not os.path.exists('attack/mnist/'):
    os.makedirs('attack/mnist/')

if len(sys.argv) < 2:
    print('Usage: python svae_mnist.py train')
    print('       python svae_mnist.py generate')
    print('       python svae_mnist.py generate test_index target_label')
    exit(0)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

f1 = MLP().to(device)

encoder, generator, classifier, discriminator = mnist_model.build_MNIST_Model(dim_z)
encoder = encoder.to(device)
generator = generator.to(device)
classifier = classifier.to(device)
discriminator = discriminator.to(device)

# For a MLP model (f1) pretrained on MNIST

for p in f1.parameters():
    p.requires_grad = False

stage1_params = itertools.chain(encoder.parameters(), generator.parameters(), classifier.parameters())
stage2_params = discriminator.parameters()

if sys.argv[1] == 'train':
# =============================== TRAINING ====================================

    # stage1_solver = optim.Adam(stage1_params, lr=lr)
    stage1_solver = optim.Adam(stage1_params, lr=lr, betas=(0.5, 0.999))
    # stage2_solver = optim.Adam(stage2_params, lr=lr)
    stage2_solver = optim.Adam(stage2_params, lr=lr, betas=(0.5, 0.999))

    best_VAE_loss = 1000.
    best_DIS_loss = 1000.

    best_epoch = -1

    x_train, y_train, x_test, y_test = mnist_data.load_mnist(reshape=True, twoclass=None, binary=True, onehot=False)

    n_train_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    total_train_batch = n_train_samples // batch_size
    total_test_batch = n_test_samples // batch_size

# =============================== Stage1 TRAINING =============================
    
    if stage_flag[0]:
        if pretrain_epoch_num[0] >= 0:
            encoder.load_state_dict(torch.load('out/mnist/encoder_%d.pth' % (pretrain_epoch_num[0])))
            generator.load_state_dict(torch.load('out/mnist/generator_%d.pth' % (pretrain_epoch_num[0])))
            classifier.load_state_dict(torch.load('out/mnist/classifier_%d.pth' % (pretrain_epoch_num[0])))
            
        for epoch in range(pretrain_epoch_num[0] + 1, epoch_num[0] + 1):
            print('Stage1 Training, Epoch %d:' % (epoch))
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
                
                X, y = torch.from_numpy(batch_xs).float().to(device), torch.from_numpy(batch_ys).long().to(device)

                stage1_solver.zero_grad()

                # Forward
                z_mu, z_sigma = encoder(X)
                z = z_mu + z_sigma * torch.randn(batch_size, dim_z).to(device)
                X_hat = generator(z)
                X_hat_clipped = torch.clamp(X_hat, EPSILON, 1 - EPSILON)
                y_hat = classifier(z)

                # Loss
                classify_loss = nn.CrossEntropyLoss()(y_hat, y)
                recon_loss = -torch.sum(X * torch.log(X_hat_clipped) + (1 - X) * torch.log(1 - X_hat_clipped)) / batch_size
                kl_loss = torch.mean(0.5 * torch.sum(z_mu ** 2 + z_sigma ** 2 - torch.log(EPSILON + z_sigma ** 2) - 1., 1))
                loss = classify_loss + recon_loss + kl_loss        

                # Backward
                loss.backward()

                # Update
                stage1_solver.step()

            with torch.no_grad():
                # validate after each epoch
                val_loss, val_cla, val_rec, val_kl, val_acc = 0., 0., 0., 0., 0.

                for i in range(total_test_batch):
                    offset = (i * batch_size) % n_test_samples
                    batch_xs = x_test[offset:(offset + batch_size), ...]
                    batch_ys = y_test[offset:(offset + batch_size), ...]

                    X, y = torch.from_numpy(batch_xs).float().to(device), torch.from_numpy(batch_ys).long().to(device)

                    z_mu, z_sigma = encoder(X)
                    z = z_mu + z_sigma * torch.randn(batch_size, dim_z).to(device)
                    X_hat = generator(z)
                    X_hat_clipped = torch.clamp(X_hat, EPSILON, 1 - EPSILON)
                    y_hat = classifier(z)

                    # Loss
                    classify_loss = nn.CrossEntropyLoss()(y_hat, y)
                    recon_loss = -torch.sum(X * torch.log(X_hat_clipped) + (1 - X) * torch.log(1 - X_hat_clipped)) / batch_size
                    kl_loss = torch.mean(0.5 * torch.sum(z_mu ** 2 + z_sigma ** 2 - torch.log(EPSILON + z_sigma ** 2) - 1., 1))
                    loss = classify_loss + recon_loss + kl_loss
                    _, predicted = torch.max(y_hat.data, 1)
                    acc = torch.mean(torch.eq(predicted, y).float())

                    val_loss += loss.item()
                    val_cla += classify_loss.item()
                    val_rec += recon_loss.item()
                    val_kl += kl_loss.item()
                    val_acc += acc.item()

                print('val_loss:{:.6}, val_cla:{:.6}, val_rec:{:.6}, val_kl:{:.6}, val_acc:{:.4}'.format(val_loss / total_test_batch,
                                                                                            val_cla / total_test_batch,
                                                                                            val_rec / total_test_batch,
                                                                                            val_kl / total_test_batch,
                                                                                            val_acc / total_test_batch))
                if (val_loss / total_test_batch) < best_VAE_loss:
                    torch.save(encoder.state_dict(), 'out/mnist/encoder_best_%d.pth' % (epoch))
                    torch.save(classifier.state_dict(), 'out/mnist/classifier_best_%d.pth' % (epoch))
                    torch.save(generator.state_dict(), 'out/mnist/generator_best_%d.pth' % (epoch))
                    best_VAE_loss = val_loss / total_test_batch

                    if best_epoch >= 0:
                        os.system('rm out/mnist/encoder_best_%d.pth' % (best_epoch))
                        os.system('rm out/mnist/classifier_best_%d.pth' % (best_epoch))
                        os.system('rm out/mnist/generator_best_%d.pth' % (best_epoch))

                    best_epoch = epoch

            if epoch % 10 == 0:
                torch.save(encoder.state_dict(), 'out/mnist/encoder_%d.pth' % (epoch))
                torch.save(classifier.state_dict(), 'out/mnist/classifier_%d.pth' % (epoch))
                torch.save(generator.state_dict(), 'out/mnist/generator_%d.pth' % (epoch))

                if epoch >= 10:
                    os.system('rm out/mnist/encoder_%d.pth' % (epoch - 10))
                    os.system('rm out/mnist/classifier_%d.pth' % (epoch - 10))
                    os.system('rm out/mnist/generator_%d.pth' % (epoch - 10))


        print('-----------------------------------------------------------------')

# =============================== Stage2 TRAINING =============================

    if stage_flag[1]:
        encoder.load_state_dict(torch.load('out/mnist/encoder_best_%d.pth'  % (best_epoch) ))
        best_epoch = -1

        for p in encoder.parameters():
            p.requires_grad = False

        if pretrain_epoch_num[1] >= 0:
            discriminator.load_state_dict(torch.load('out/mnist/discriminator_%d.pth' % (pretrain_epoch_num[1])))

        for epoch in range(epoch_num[1] + 1):
            print('Stage2 Training, Epoch %d:' % (epoch))
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
                
                X, y = torch.from_numpy(batch_xs).float().to(device), torch.from_numpy(batch_ys).long().to(device)

                stage2_solver.zero_grad()

                # Forward
                z_mu, z_sigma = encoder(X)
                z_real = z_mu + z_sigma * torch.randn(batch_size, dim_z).to(device)
                z_fake = torch.randn(batch_size, dim_z).to(device)

                D_real = discriminator(z_real)
                D_fake = discriminator(z_fake)

                # Loss
                loss = -torch.mean(torch.log(torch.sigmoid(D_real)) + torch.log(1 - torch.sigmoid(D_fake)))

                # Backward
                loss.backward()

                # Update
                stage2_solver.step()

            with torch.no_grad():
                # validate after each epoch
                val_loss_dis, acc_dis_true, acc_dis_fake = 0., 0., 0.

                for i in range(total_test_batch):
                    offset = (i * batch_size) % n_test_samples
                    batch_xs = x_test[offset:(offset + batch_size), ...]
                    batch_ys = y_test[offset:(offset + batch_size), ...]

                    X, y = torch.from_numpy(batch_xs).float().to(device), torch.from_numpy(batch_ys).long().to(device)

                    z_mu, z_sigma = encoder(X)
                    z_real = z_mu + z_sigma * torch.randn(batch_size, dim_z).to(device)
                    z_fake = torch.randn(batch_size, dim_z).to(device)

                    D_real = discriminator(z_real)
                    D_fake = discriminator(z_fake)

                    loss = -torch.mean(torch.log(torch.sigmoid(D_real)) + torch.log(1 - torch.sigmoid(D_fake)))
                   
                    val_loss_dis += loss.item()
                    acc_dis_true += torch.mean(torch.ge(D_real, 0.5).float()).item()
                    acc_dis_fake += torch.mean(torch.lt(D_fake, 0.5).float()).item()

                print('val_loss_dis:{:.6}, acc_dis_true:{:.4}, acc_dis_fake:{:.4}'.format(val_loss_dis / total_test_batch,
                                                                                               acc_dis_true / total_test_batch,
                                                                                               acc_dis_fake / total_test_batch))
                if (val_loss_dis / total_test_batch) < best_DIS_loss:
                    torch.save(discriminator.state_dict(), 'out/mnist/discriminator_best_%d.pth' % (epoch))
                    best_DIS_loss = val_loss_dis / total_test_batch

                    if best_epoch >= 0:
                        os.system('rm out/mnist/discriminator_best_%d.pth' % (best_epoch))

                    best_epoch = epoch

            if epoch % 10 == 0:
                torch.save(discriminator.state_dict(), 'out/mnist/discriminator_%d.pth' % (epoch))

                if epoch >= 10:
                    os.system('rm out/mnist/discriminator_%d.pth' % (epoch - 10))

        print('-----------------------------------------------------------------')

        torch.save(discriminator.state_dict(), 'out/mnist/discriminator.pth')

elif sys.argv[1] == 'generate':
    encoder.load_state_dict(torch.load('out/mnist/encoder_best_551.pth'))
    generator.load_state_dict(torch.load('out/mnist/generator_best_551.pth'))
    classifier.load_state_dict(torch.load('out/mnist/classifier_best_551.pth'))
    discriminator.load_state_dict(torch.load('out/mnist/discriminator_best_544.pth'))

    f1.load_state_dict(torch.load('out/mnist/f1_best_178.pth'))

    for p in encoder.parameters():
        p.requires_grad = False

    for p in generator.parameters():
        p.requires_grad = False

    for p in classifier.parameters():
        p.requires_grad = False

    for p in discriminator.parameters():
        p.requires_grad = False

    for p in f1.parameters():
        p.requires_grad = False

    x_train, y_train, x_test, y_test = mnist_data.load_mnist(reshape=True, twoclass=None, binary=True, onehot=False)

    if len(sys.argv) > 3:
        test_img_index = int(sys.argv[2])
        target_img_label = int(sys.argv[3])
        
        test_label = y_test[test_img_index]

        test_img = x_test[test_img_index, ...]
        true_label = np.zeros(shape=[1, ])
        true_label[0] = test_label

        print('Original Label: %d' % (test_label))

        target_label = np.zeros(shape=[1, ])
        target_label[0] = target_img_label

        print('Target Label: %d' % (target_img_label))
        
        img = np.repeat(test_img.transpose((1, 2, 0)), 3, axis=2)
        plt.imsave('attack/mnist/test.png', img)

        X, y = torch.from_numpy(np.expand_dims(test_img, 0)).float().to(device), torch.from_numpy(true_label).long().to(device)
        y_hat = torch.from_numpy(target_label).long().to(device)

        z, _ = encoder(X)
        z = Variable(z, requires_grad=True).to(device)
        
        z_solver = optim.Adam([z], lr=z_lr)

        k = 0
        iter_num = 20000
        
        for it in range(iter_num + 1):
            z_solver.zero_grad()

            # Forward
            y2 = classifier(z)
            D = discriminator(z)
            X_hat = generator(z)
            y1 = f1(X_hat)

            # loss
            J1 = nn.CrossEntropyLoss()(y1, y)
            J2 = nn.CrossEntropyLoss()(y2, y_hat)
            J_IT = J2 + 0.01 * torch.mean(1 - torch.sigmoid(D)) + 0.0001 * torch.mean(torch.norm(z, dim=1))
            J_SA = J_IT + k * J1
            k = k + z_lr * (0.001 * J1.item() - J2.item() + max(J1.item() - J1_hat, 0))
            k = max(0, min(k, 0.005))
            
            if (it % 1000 == 0):
                print('iter-%d: J_SA: %.6f, J_IT: %.6f, J1: %.6f' % (it, J_SA.item(), J_IT.item(), J1.item()))
                img = torch.squeeze(X_hat).data.cpu().numpy()
                img = np.repeat(img[..., np.newaxis], 3, axis=2)
                plt.imsave('attack/mnist/iter-%d.png' % (it), img)

            # Backward
            J_SA.backward()

            # Update
            z_solver.step()
    
    else:
        if not os.path.exists('attack/mnist/data/'):
            os.makedirs('attack/mnist/data/')

        for label in range(classes):
            if not os.path.exists('attack/mnist/data/' + str(label) + '/'):
                os.makedirs('attack/mnist/data/' + str(label) + '/')

        n_train_samples = x_train.shape[0]
        total_train_batch = n_train_samples // batch_size

        for i in range(total_train_batch):
            # Compute the offset of the current minibatch in the data.
            offset = (i * batch_size) % n_train_samples
            batch_test_img = x_train[offset:(offset + batch_size), ...]
            batch_test_label = y_train[offset:(offset + batch_size), ...]

            print('Batch %d, Original Labels:' % (i))
            print(batch_test_label.tolist())

            batch_target_label = (batch_test_label + 1) % classes

            print('Batch %d, Target Labels:' % (i))
            print(batch_target_label.tolist())

            X, y = torch.from_numpy(batch_test_img).float().to(device), torch.from_numpy(batch_test_label).long().to(device)
            y_hat = torch.from_numpy(batch_target_label).long().to(device)

            z, _ = encoder(X)
            z = Variable(z, requires_grad=True).to(device)
            
            z_solver = optim.Adam([z], lr=z_lr)
            
            k = 0
            iter_num = 20000
            
            for it in range(iter_num + 1):
                z_solver.zero_grad()

                # Forward
                y2 = classifier(z)
                D = discriminator(z)
                X_hat = generator(z)
                y1 = f1(X_hat)

                # loss
                J1 = nn.CrossEntropyLoss()(y1, y)
                J2 = nn.CrossEntropyLoss()(y2, y_hat)
                J_IT = J2 + 0.01 * torch.mean(1 - torch.sigmoid(D)) + 0.0001 * torch.mean(torch.norm(z, dim=1))
                J_SA = J_IT + k * J1
                k = k + z_lr * (0.001 * J1.item() - J2.item() + max(J1.item() - J1_hat, 0))
                k = max(0, min(k, 0.005))
                
                '''
                if (it % 1000 == 0):
                    print('iter-%d: J_SA: %.6f, J_IT: %.6f, J1: %.6f' % (it, J_SA.item(), J_IT.item(), J1.item()))
                '''
                
                if (it == iter_num):
                    samples = X_hat.permute(0, 2, 3, 1).data.cpu().numpy()

                    for ind in range(batch_size):
                        img = np.repeat(samples[ind], 3, axis=2)
                        plt.imsave('attack/mnist/data/%d/%d.png' % (batch_target_label[ind], i * batch_size + ind), img)


                # Backward
                J_SA.backward()

                # Update
                z_solver.step()
    
else:
    print('Usage: python svae_mnist.py train')
    print('       python svae_mnist.py generate')
    print('       python svae_mnist.py generate test_index target_label')
    exit(0)