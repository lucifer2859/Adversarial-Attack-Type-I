import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, "/home/dchen/SVAE/classification-cifar10-pytorch/")

import cifar_model_v1
import cifar_model_v2
import cifar_data
from models import *
import itertools

import pytorch_msssim

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

EPSILON = 1e-6

epoch_num = [600, 600]
pretrain_epoch_num = [-1, -1]
stage_flag = [True, True]
batch_size = 64
dim_z = 1024
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
lr = 0.0002
z_lr = 0.005
J1_hat = 0.05

transform_train = transforms.Compose([
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

if not os.path.exists('out/'):
    os.makedirs('out/')

if not os.path.exists('out/cifar/'):
    os.makedirs('out/cifar/')

if not os.path.exists('attack/'):
    os.makedirs('attack/')

if not os.path.exists('attack/cifar/'):
    os.makedirs('attack/cifar/')

if len(sys.argv) < 2:
    print('Usage: python svae_cifar.py train')
    print('       python svae_cifar.py generate')
    print('       python svae_cifar.py generate test_index')
    print('       python svae_cifar.py generate test_index target_label')
    exit(0)

# f1 =  MobileNetV2()
f1 = DPN92()
net_name = f1.name
save_path = 'out/cifar/f1_{0}.pth'.format(f1.name)
f1 = f1.to(device)

checkpoint = torch.load(save_path)
f1.load_state_dict(checkpoint['net'])

encoder, generator, classifier, discriminator = cifar_model_v1.build_CIFAR_Model(dim_z, device=device)
# encoder, generator, classifier, discriminator = cifar_model_v2.build_CIFAR_Model(dim_z, device=device)

# For a MobileNetV2 model (f1) pretrained on CIFAR-10

for p in f1.parameters():
    p.requires_grad = False

stage1_params = itertools.chain(encoder.parameters(), generator.parameters(), classifier.parameters())
stage2_params = discriminator.parameters()

if sys.argv[1] == 'train':
# =============================== TRAINING ====================================

    trainset = torchvision.datasets.CIFAR10(root='/home/dchen/dataset', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='/home/dchen/dataset', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    stage1_solver = optim.Adam(stage1_params, lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
    stage2_solver = optim.Adam(stage2_params, lr=lr, betas=(0.5, 0.999))

    best_VAE_loss = 10000.
    best_DIS_loss = 10000.

    best_epoch = -1

# =============================== Stage1 TRAINING =============================
    
    if stage_flag[0]:
        if pretrain_epoch_num[0] >= 0:
            encoder.load_state_dict(torch.load('out/cifar/encoder_%d.pth' % (pretrain_epoch_num[0])))
            generator.load_state_dict(torch.load('out/cifar/generator_%d.pth' % (pretrain_epoch_num[0])))
            classifier.load_state_dict(torch.load('out/cifar/classifier_%d.pth' % (pretrain_epoch_num[0])))
            
        for epoch in range(pretrain_epoch_num[0] + 1, epoch_num[0] + 1):
            print('Stage1 Training, Epoch %d:' % (epoch))
            
            for X, y in trainloader:
                X, y = X.to(device), y.to(device)

                stage1_solver.zero_grad()

                # Forward
                z_mu, z_sigma = encoder(X)
                z = z_mu + z_sigma * torch.randn(X.size()[0], dim_z).to(device)
                X_hat = generator(z)
                y_hat = classifier(z)

                # Loss
                classify_loss = nn.CrossEntropyLoss()(y_hat, y)
                mse_loss = nn.MSELoss(reduction='sum')(X_hat, X) / X.size()[0]
                msssim_loss = 1. - pytorch_msssim.msssim(X_hat, X)
                kl_loss = torch.mean(0.5 * torch.sum(z_mu ** 2 + z_sigma ** 2 - torch.log(EPSILON + z_sigma ** 2) - 1., 1))
                
                loss = None

                if (msssim_loss.item() < 0.4):
                    loss = classify_loss + msssim_loss + kl_loss        
                else:
                    loss = classify_loss + mse_loss + kl_loss        
                
                loss.backward()

                # Update
                stage1_solver.step()

            with torch.no_grad():
                # validate after each epoch
                val_loss, val_cla, val_mse, val_msssim, val_kl, val_acc = 0., 0., 0., 0., 0., 0.

                for X, y in testloader:
                    X, y = X.to(device), y.to(device)

                    z_mu, z_sigma = encoder(X)
                    z = z_mu + z_sigma * torch.randn(X.size()[0], dim_z).to(device)
                    X_hat = generator(z)
                    y_hat = classifier(z)

                    # Loss
                    classify_loss = nn.CrossEntropyLoss()(y_hat, y)
                    mse_loss = nn.MSELoss(reduction='sum')(X_hat, X) / X.size()[0]
                    msssim_loss = 1. - pytorch_msssim.msssim(X_hat, X)
                    kl_loss = torch.mean(0.5 * torch.sum(z_mu ** 2 + z_sigma ** 2 - torch.log(EPSILON + z_sigma ** 2) - 1., 1))
                    
                    loss = None

                    if (msssim_loss.item() < 0.4):
                        loss = classify_loss + msssim_loss + kl_loss        
                    else:
                        loss = classify_loss + mse_loss + kl_loss 

                    _, predicted = torch.max(y_hat.data, 1)
                    acc = torch.mean(torch.eq(predicted, y).float())

                    val_loss += loss.item()
                    val_cla += classify_loss.item()
                    val_mse += mse_loss.item()
                    val_msssim += msssim_loss.item()
                    val_kl += kl_loss.item()
                    val_acc += acc.item()

                print('val_loss:{:.6}, val_cla:{:.6}, val_mse:{:.6}, val_msssim:{:.6}, val_kl:{:.6}, val_acc:{:.4}'.format(val_loss / len(testloader),
                                                                                            val_cla / len(testloader),
                                                                                            val_mse / len(testloader),
                                                                                            val_msssim / len(testloader),
                                                                                            val_kl / len(testloader),
                                                                                            val_acc / len(testloader)))
                if (val_loss / len(testloader)) < best_VAE_loss:
                    torch.save(encoder.state_dict(), 'out/cifar/encoder_best_%d.pth' % (epoch))
                    torch.save(classifier.state_dict(), 'out/cifar/classifier_best_%d.pth' % (epoch))
                    torch.save(generator.state_dict(), 'out/cifar/generator_best_%d.pth' % (epoch))
                    best_VAE_loss = val_loss / len(testloader)

                    if best_epoch >= 0:
                        os.system('rm out/cifar/encoder_best_%d.pth' % (best_epoch))
                        os.system('rm out/cifar/classifier_best_%d.pth' % (best_epoch))
                        os.system('rm out/cifar/generator_best_%d.pth' % (best_epoch))

                    best_epoch = epoch

            if epoch % 10 == 0:
                torch.save(encoder.state_dict(), 'out/cifar/encoder_%d.pth' % (epoch))
                torch.save(classifier.state_dict(), 'out/cifar/classifier_%d.pth' % (epoch))
                torch.save(generator.state_dict(), 'out/cifar/generator_%d.pth' % (epoch))

                if epoch >= 10:
                    os.system('rm out/cifar/encoder_%d.pth' % (epoch - 10))
                    os.system('rm out/cifar/classifier_%d.pth' % (epoch - 10))
                    os.system('rm out/cifar/generator_%d.pth' % (epoch - 10))


        print('-----------------------------------------------------------------')

# =============================== Stage2 TRAINING =============================

    if stage_flag[1]:
        encoder.load_state_dict(torch.load('out/cifar/encoder_best_%d.pth'  % (best_epoch) ))
        best_epoch = -1

        for p in encoder.parameters():
            p.requires_grad = False

        if pretrain_epoch_num[1] >= 0:
            discriminator.load_state_dict(torch.load('out/cifar/discriminator_%d.pth' % (pretrain_epoch_num[1])))

        for epoch in range(epoch_num[1] + 1):
            print('Stage2 Training, Epoch %d:' % (epoch))
            
            for X, y in trainloader:
                X, y = X.to(device), y.to(device)

                stage2_solver.zero_grad()

                # Forward
                z_mu, z_sigma = encoder(X)
                z_real = z_mu + z_sigma * torch.randn(X.size()[0], dim_z).to(device)
                z_fake = torch.randn(X.size()[0], dim_z).to(device)

                D_real = discriminator(z_real)
                D_fake = discriminator(z_fake)

                # Loss
                loss = -torch.mean(torch.log(torch.sigmoid(D_real)) + torch.log(1 - torch.sigmoid(D_fake)))

                if (torch.isnan(loss)):
                    loss.zero_()

                # Backward
                loss.backward()

                # Update
                stage2_solver.step()

            with torch.no_grad():
                # validate after each epoch
                val_loss_dis, acc_dis_true, acc_dis_fake = 0., 0., 0.

                for X, y in testloader:
                    X, y = X.to(device), y.to(device)

                    z_mu, z_sigma = encoder(X)
                    z_real = z_mu + z_sigma * torch.randn(X.size()[0], dim_z).to(device)
                    z_fake = torch.randn(X.size()[0], dim_z).to(device)

                    D_real = discriminator(z_real)
                    D_fake = discriminator(z_fake)

                    loss = -torch.mean(torch.log(torch.sigmoid(D_real)) + torch.log(1 - torch.sigmoid(D_fake)))
                   
                    val_loss_dis += loss.item()
                    acc_dis_true += torch.mean(torch.ge(D_real, 0.5).float()).item()
                    acc_dis_fake += torch.mean(torch.lt(D_fake, 0.5).float()).item()

                print('val_loss_dis:{:.6}, acc_dis_true:{:.4}, acc_dis_fake:{:.4}'.format(val_loss_dis / len(testloader),
                                                                                               acc_dis_true / len(testloader),
                                                                                               acc_dis_fake / len(testloader)))
                if (val_loss_dis / len(testloader)) < best_DIS_loss:
                    torch.save(discriminator.state_dict(), 'out/cifar/discriminator_best_%d.pth' % (epoch))
                    best_DIS_loss = val_loss_dis / len(testloader)

                    if best_epoch >= 0:
                        os.system('rm out/cifar/discriminator_best_%d.pth' % (best_epoch))

                    best_epoch = epoch

            if epoch % 10 == 0:
                torch.save(discriminator.state_dict(), 'out/cifar/discriminator_%d.pth' % (epoch))

                if epoch >= 10:
                    os.system('rm out/cifar/discriminator_%d.pth' % (epoch - 10))

        print('-----------------------------------------------------------------')

elif sys.argv[1] == 'generate':
    encoder.load_state_dict(torch.load('out/cifar/encoder_best_75.pth'))
    generator.load_state_dict(torch.load('out/cifar/generator_best_75.pth'))
    classifier.load_state_dict(torch.load('out/cifar/classifier_best_75.pth'))
    discriminator.load_state_dict(torch.load('out/cifar/discriminator_best_553.pth'))

    mean = torch.Tensor([0.4914, 0.4822, 0.4465]).reshape((1, 3, 1, 1)).to(device)
    std = torch.Tensor([0.2023, 0.1994, 0.2010]).reshape((1, 3, 1, 1)).to(device)

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

    if len(sys.argv) > 3:
        test_img_index = int(sys.argv[2])
        target_img_label = int(sys.argv[3])

        testset = torchvision.datasets.CIFAR10(root='/home/dchen/dataset', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

        iter_num = 20000

        for i, (X, y) in enumerate(testloader):
            if i != test_img_index:
                continue

            X, y = X.to(device), y.to(device)

            img = torch.squeeze(X).permute(1, 2, 0).data.cpu().numpy()
            plt.imsave('attack/cifar/test.png', img)

            y_hat = torch.from_numpy(np.array([target_img_label])).long().to(device)

            print('Original Label: %s' % (classes[y.data.cpu().numpy()[0]]))
            print('Target Label: %s' % (classes[target_img_label]))

            z, _ = encoder(X)
            z = Variable(z, requires_grad=True).to(device)
                
            z_solver = optim.Adam([z], lr=z_lr)
                
            k = 0
                
            for it in range(iter_num + 1):
                z_solver.zero_grad()

                # Forward
                y2 = classifier(z)
                D = discriminator(z)
                X_hat = generator(z)
                X_hat_norm = (X_hat - mean) / std
                y1 = f1(X_hat_norm)

                # loss
                J1 = nn.CrossEntropyLoss()(y1, y)
                J2 = nn.CrossEntropyLoss()(y2, y_hat)
                J_IT = J2 + 0.01 * torch.mean(1 - torch.sigmoid(D)) + 0.0001 * torch.mean(torch.norm(z, dim=1))
                J_SA = J_IT + k * J1
                k = k + z_lr * (0.001 * J1.item() - J2.item() + max(J1.item() - J1_hat, 0))
                k = max(0, min(k, 0.005))
                    

                if (it % 1000 == 0):
                    print('iter-%d: J_SA: %.6f, J_IT: %.6f, J1: %.6f' % (it, J_SA.item(), J_IT.item(), J1.item()))
                    img = torch.squeeze(X_hat).permute(1, 2, 0).data.cpu().numpy()

                    plt.imsave('attack/cifar/iter-%d.png' % (it), img)

                # Backward
                J_SA.backward()

                # Update
                z_solver.step()

            break

    else:
        testset = torchvision.datasets.CIFAR10(root='/home/dchen/dataset', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        if not os.path.exists('attack/cifar/data/'):
            os.makedirs('attack/cifar/data/')

        for label in classes:
            if not os.path.exists('attack/cifar/data/' + label + '/'):
                os.makedirs('attack/cifar/data/' + label + '/')

        iter_num = 20000

        for i, (X, y) in enumerate(testloader):
            X, y = X.to(device), y.to(device)

            y_hat = (y + 1) % len(classes)

            batch_target_label = y_hat.data.cpu().numpy()

            z, _ = encoder(X)
            z = Variable(z, requires_grad=True).to(device)
                
            z_solver = optim.Adam([z], lr=z_lr)
                
            k = 0
                
            for it in range(iter_num + 1):
                z_solver.zero_grad()

                # Forward
                y2 = classifier(z)
                D = discriminator(z)
                X_hat = generator(z)
                X_hat_norm = (X_hat - mean) / std
                y1 = f1(X_hat_norm)

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

                    for ind in range(X.size()[0]):
                        plt.imsave('attack/cifar/data/%s/%d.png' % (classes[batch_target_label[ind]], i * batch_size + ind), samples[ind])

                # Backward
                J_SA.backward()

                # Update
                z_solver.step()

elif sys.argv[1] == 'generateNoise':
    encoder.load_state_dict(torch.load('out/cifar/encoder_best_13.pth'))
    generator.load_state_dict(torch.load('out/cifar/generator_best_13.pth'))
    classifier.load_state_dict(torch.load('out/cifar/classifier_best_13.pth'))

    mean = torch.Tensor([0.4914, 0.4822, 0.4465]).reshape((1, 3, 1, 1)).to(device)
    std = torch.Tensor([0.2023, 0.1994, 0.2010]).reshape((1, 3, 1, 1)).to(device)

    for p in encoder.parameters():
        p.requires_grad = False

    for p in generator.parameters():
        p.requires_grad = False

    for p in classifier.parameters():
        p.requires_grad = False

    for p in f1.parameters():
        p.requires_grad = False

    if len(sys.argv) > 2:
        test_img_index = int(sys.argv[2])

        testset = torchvision.datasets.CIFAR10(root='/home/dchen/dataset', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

        iter_num = 20000

        for i, (X, y) in enumerate(testloader):
            if i != test_img_index:
                continue

            X, y = X.to(device), y.to(device)

            img = torch.squeeze(X).permute(1, 2, 0).data.cpu().numpy()
            plt.imsave('attack/cifar/test.png', img)

            print('Label: %s' % (classes[y.data.cpu().numpy()[0]]))

            z, _ = encoder(X)
            z_ori = z.clone().detach()
            z = Variable(z, requires_grad=True).to(device)
                
            z_solver = optim.Adam([z], lr=z_lr)
                
            k = 0
                
            for it in range(iter_num + 1):
                z_solver.zero_grad()

                # Forward
                y2 = classifier(z)
                X_hat = generator(z)
                X_hat_norm = (X_hat - mean) / std
                y1 = f1(X_hat_norm)

                # loss
                J1 = nn.CrossEntropyLoss()(y1, y)
                # J2 = F.relu(2. - torch.mean(torch.norm(z - z_ori, p=1, dim=1)) / z.size()[1])
                J2 = F.relu(2. - nn.MSELoss(size_average=True)(X_hat, X))
                J_SA = J2 + k * J1
                k = k + z_lr * (0.001 * J1.item() - J2.item() + max(J1.item() - J1_hat, 0))
                k = max(0, min(k, 0.005))

                if (it % 1000 == 0):
                    dev = torch.mean(torch.norm((z - z_ori) / z_ori, dim=1)) / z.size()[1]
                    print('iter-%d: J_SA: %.6f, J1: %.6f, J2: %.6f, dev: %.6f' % (it, J_SA.item(), J1.item(), J2.item(), dev.item()))
                    img = torch.squeeze(X_hat).permute(1, 2, 0).data.cpu().numpy()

                    plt.imsave('attack/cifar/iter-%d.png' % (it), img)

                # Backward
                J_SA.backward()

                # Update
                z_solver.step()

            break
    
else:
    print('Usage: python svae_cifar.py train')
    print('       python svae_cifar.py generate')
    print('       python svae_cifar.py generate test_index target_label')
    print('       python svae_cifar.py generateNoise')
    print('       python svae_cifar.py generateNoise test_index')
    exit(0)