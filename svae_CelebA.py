import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import ssl
import sys

import CelebA_model
import CelebA_data
import itertools
from PIL import Image
from facenet_pytorch import InceptionResnetV1

ssl._create_default_https_context = ssl._create_unverified_context

np.random.seed(996)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

EPSILON = 1e-6

IMAGE_SIZE = 64

epoch_num = [600, 600]
pretrain_epoch_num = [-1, -1]
stage_flag = [True, True]
batch_size = 64
dim_z = 1024
lr = 0.0002
z_lr = 0.005
J1_hat = 1.00

if not os.path.exists('out/'):
    os.makedirs('out/')

if not os.path.exists('out/CelebA/'):
    os.makedirs('out/CelebA/')

if not os.path.exists('attack/'):
    os.makedirs('attack/')

if not os.path.exists('attack/CelebA/'):
    os.makedirs('attack/CelebA/')

if len(sys.argv) < 2:
    print('Usage: python svae_CelebA.py train')
    print('       python svae_CelebA.py generate test_index')
    exit(0)

encoder, generator, classifier, discriminator = CelebA_model.build_CelebA_Model(dim_z)
encoder = encoder.to(device)
generator = generator.to(device)
classifier = classifier.to(device)
discriminator = discriminator.to(device)

# For a FaceNet model pretrained on VGGFace2 
facenet = InceptionResnetV1(num_classes=8631).eval()
# facenet.load_state_dict(torch.load('/home/dchen/GM/pretrain_models/20180402-114759-vggface2.pt', map_location=device))
facenet.load_state_dict(torch.load('20180402-114759-vggface2.pt', map_location=device))
facenet = facenet.to(device)

for p in facenet.parameters():
    p.requires_grad = False


def get_string(label):
    if label == 0:
        return 'female'
    else:
        return 'male'

def distance(x1, x2):
    return torch.mean(torch.norm(x1 - x2, dim=1))


stage1_params = itertools.chain(encoder.parameters(), generator.parameters(), classifier.parameters())
stage2_params = discriminator.parameters()

if sys.argv[1] == 'train':
# =============================== TRAINING ====================================

    # stage1_solver = optim.Adam(stage1_params, lr=lr)
    stage1_solver = optim.Adam(stage1_params, lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    # stage2_solver = optim.Adam(stage2_params, lr=lr)
    stage2_solver = optim.Adam(stage2_params, lr=lr, betas=(0.5, 0.999))

    best_VAE_loss = 1000.
    best_DIS_loss = 1000.

    best_epoch = -1

    print('loading celebA dataset ...')
    x_train, y_train, x_test, y_test = CelebA_data.load_celebA_Gender(data_dir='/home/dchen/dataset/CelebA/GenderSplit')
    print('done!')

    n_train_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    total_train_batch = n_train_samples // batch_size
    total_test_batch = n_test_samples // batch_size

# =============================== Stage1 TRAINING =============================
    
    if stage_flag[0]:
        if pretrain_epoch_num[0] >= 0:
            encoder.load_state_dict(torch.load('out/CelebA/encoder_%d.pth' % (pretrain_epoch_num[0])))
            generator.load_state_dict(torch.load('out/CelebA/generator_%d.pth' % (pretrain_epoch_num[0])))
            classifier.load_state_dict(torch.load('out/CelebA/classifier_%d.pth' % (pretrain_epoch_num[0])))
            
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
                
                X, y = torch.from_numpy(batch_xs).float().to(device), torch.from_numpy(batch_ys).float().to(device)

                stage1_solver.zero_grad()

                # Forward
                z_mu, z_sigma = encoder(X)
                z = z_mu + z_sigma * torch.randn(batch_size, dim_z).to(device)
                X_hat = generator(z)
                y_hat = classifier(z)

                # Loss
                classify_loss = F.binary_cross_entropy(torch.sigmoid(y_hat), y)
                recon_loss = nn.MSELoss(reduction='sum')(X_hat, X) / batch_size
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

                    X, y = torch.from_numpy(batch_xs).float().to(device), torch.from_numpy(batch_ys).float().to(device)

                    z_mu, z_sigma = encoder(X)
                    z = z_mu + z_sigma * torch.randn(batch_size, dim_z).to(device)
                    X_hat = generator(z)
                    y_hat = classifier(z)

                    # Loss
                    classify_loss = F.binary_cross_entropy(torch.sigmoid(y_hat), y)
                    recon_loss = nn.MSELoss(reduction='sum')(X_hat, X) / batch_size
                    kl_loss = torch.mean(0.5 * torch.sum(z_mu ** 2 + z_sigma ** 2 - torch.log(EPSILON + z_sigma ** 2) - 1., 1))
                    loss = classify_loss + recon_loss + kl_loss
                    acc = torch.mean(torch.eq(torch.round(torch.sigmoid(y_hat)), y).float())

                    val_loss += loss.item()
                    val_cla += classify_loss.item()
                    val_rec += recon_loss.item()
                    val_kl += kl_loss.item()
                    val_acc += acc.item()

                print('val_loss:{:.4}, val_cla:{:.4}, val_rec:{:.4}, val_kl:{:.4}, val_acc:{:.4}'.format(val_loss / total_test_batch,
                                                                                            val_cla / total_test_batch,
                                                                                            val_rec / total_test_batch,
                                                                                            val_kl / total_test_batch,
                                                                                            val_acc / total_test_batch))
                if (val_loss / total_test_batch) < best_VAE_loss:
                    torch.save(encoder.state_dict(), 'out/CelebA/encoder_best_%d.pth' % (epoch))
                    torch.save(classifier.state_dict(), 'out/CelebA/classifier_best_%d.pth' % (epoch))
                    torch.save(generator.state_dict(), 'out/CelebA/generator_best_%d.pth' % (epoch))
                    best_VAE_loss = val_loss / total_test_batch

                    if best_epoch >= 0:
                        os.system('rm out/CelebA/encoder_best_%d.pth' % (best_epoch))
                        os.system('rm out/CelebA/classifier_best_%d.pth' % (best_epoch))
                        os.system('rm out/CelebA/generator_best_%d.pth' % (best_epoch))

                    best_epoch = epoch

            if epoch % 10 == 0:
                torch.save(encoder.state_dict(), 'out/CelebA/encoder_%d.pth' % (epoch))
                torch.save(classifier.state_dict(), 'out/CelebA/classifier_%d.pth' % (epoch))
                torch.save(generator.state_dict(), 'out/CelebA/generator_%d.pth' % (epoch))

                if epoch >= 10:
                    os.system('rm out/CelebA/encoder_%d.pth' % (epoch - 10))
                    os.system('rm out/CelebA/classifier_%d.pth' % (epoch - 10))
                    os.system('rm out/CelebA/generator_%d.pth' % (epoch - 10))


        print('-----------------------------------------------------------------')
        
        torch.save(encoder.state_dict(), 'out/CelebA/encoder.pth')
        torch.save(classifier.state_dict(), 'out/CelebA/classifier.pth')
        torch.save(generator.state_dict(), 'out/CelebA/generator.pth')

# =============================== Stage2 TRAINING =============================

    if stage_flag[1]:
        best_epoch = -1
        
        encoder.load_state_dict(torch.load('out/CelebA/encoder.pth'))

        for p in encoder.parameters():
            p.requires_grad = False

        if pretrain_epoch_num[1] >= 0:
            discriminator.load_state_dict(torch.load('out/CelebA/discriminator_%d.pth' % (pretrain_epoch_num[1])))

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
                
                X, y = torch.from_numpy(batch_xs).float().to(device), torch.from_numpy(batch_ys).float().to(device)

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

                    X, y = torch.from_numpy(batch_xs).float().to(device), torch.from_numpy(batch_ys).float().to(device)

                    z_mu, z_sigma = encoder(X)
                    z_real = z_mu + z_sigma * torch.randn(batch_size, dim_z).to(device)
                    z_fake = torch.randn(batch_size, dim_z).to(device)

                    D_real = discriminator(z_real)
                    D_fake = discriminator(z_fake)

                    loss = -torch.mean(torch.log(torch.sigmoid(D_real)) + torch.log(1 - torch.sigmoid(D_fake)))
                   
                    val_loss_dis += loss.item()
                    acc_dis_true += torch.mean(torch.ge(D_real, 0.5).float()).item()
                    acc_dis_fake += torch.mean(torch.lt(D_fake, 0.5).float()).item()

                print('val_loss_dis:{:.4}, acc_dis_true:{:.4}, acc_dis_fake:{:.4}'.format(val_loss_dis / total_test_batch,
                                                                                               acc_dis_true / total_test_batch,
                                                                                               acc_dis_fake / total_test_batch))
                if (val_loss_dis / total_test_batch) < best_DIS_loss:
                    torch.save(discriminator.state_dict(), 'out/CelebA/discriminator_best_%d.pth' % (epoch))
                    best_DIS_loss = val_loss_dis / total_test_batch

                    if best_epoch >= 0:
                        os.system('rm out/CelebA/discriminator_best_%d.pth' % (best_epoch))

                    best_epoch = epoch

            if epoch % 10 == 0:
                torch.save(discriminator.state_dict(), 'out/CelebA/discriminator_%d.pth' % (epoch))

                if epoch >= 10:
                    os.system('rm out/CelebA/discriminator_%d.pth' % (epoch - 10))

        print('-----------------------------------------------------------------')

        torch.save(discriminator.state_dict(), 'out/CelebA/discriminator.pth')

elif sys.argv[1] == 'generate':
    test_img_index = int(sys.argv[2])
    _, _, x_test, y_test = CelebA_data.load_celebA_Gender(data_dir='/home/dchen/dataset/CelebA/GenderSplit', test_only=True)

    # choose test image and its target label
    test_img = x_test[test_img_index, ...]
    test_label = y_test[test_img_index, ...]
    test_img_label = test_label.squeeze()
    print('original_label is :', get_string(test_img_label))
    target_img_label = 1 - test_img_label
    target_label = np.zeros(shape=[1, 1])
    target_label[0, 0] = target_img_label
    print("target_label is ", get_string(target_img_label))
    
    img = test_img.transpose((1,2,0))
    plt.imsave('attack/CelebA/test.png', img)

    encoder.load_state_dict(torch.load('out/CelebA/encoder.pth'))
    generator.load_state_dict(torch.load('out/CelebA/generator.pth'))
    classifier.load_state_dict(torch.load('out/CelebA/classifier.pth'))
    discriminator.load_state_dict(torch.load('out/CelebA/discriminator.pth'))

    for p in encoder.parameters():
        p.requires_grad = False

    for p in generator.parameters():
        p.requires_grad = False

    for p in classifier.parameters():
        p.requires_grad = False

    for p in discriminator.parameters():
        p.requires_grad = False

    X, y = torch.from_numpy(np.expand_dims(test_img, 0)).float().to(device), torch.from_numpy(np.expand_dims(test_label, 0)).float().to(device)
    y_hat = torch.from_numpy(target_label).float().to(device)

    X_embedding= facenet(F.interpolate(X, size=(160, 160), mode='bilinear', align_corners=True))

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

        X_hat_embedding = facenet(F.interpolate(X_hat, size=(160, 160), mode='bilinear', align_corners=True))

        # loss
        J1 = distance(X_embedding, X_hat_embedding)
        J2 = F.binary_cross_entropy(torch.sigmoid(y2), y_hat)
        J_IT = J2 + 0.01 * torch.mean(1 - torch.sigmoid(D)) + 0.0001 * torch.mean(torch.norm(z, dim=1))
        J_SA = J_IT + k * J1
        k += z_lr * (0.001 * J1.item() - J2.item() + max(J1.item() - J1_hat, 0))
        k = max(0, min(k, 0.005))
        
        if (it % 1000 == 0):
            print('iter-%d: J_SA: %.4f, J_IT: %.4f, J1: %.4f' % (it, J_SA.item(), J_IT.item(), J1.item()))
            img = torch.squeeze(X_hat).permute(1, 2, 0).data.cpu().numpy()
            plt.imsave('attack/CelebA/iter-%d.png' % (it), img)

        # Backward
        J_SA.backward()

        # Update
        z_solver.step()
    
else:
    print('Usage: python svae_CelebA.py train')
    print('       python svae_CelebA.py generate test_index')
    exit(0)