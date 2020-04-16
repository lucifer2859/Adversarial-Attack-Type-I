import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
import ssl
import sys

import warnings
warnings.filterwarnings("ignore")

# [8, 5, 9, 0, 4, 7, 5, 0, 4, 4, 7, 3, 3, 2, 5, 0]

ssl._create_default_https_context = ssl._create_unverified_context

mnist = input_data.read_data_sets('../../MNIST', one_hot=True)
mb_size = 64 # mini-batch size
Z_dim = 100 # latent variables dim
X_dim = mnist.train.images.shape[1] # X data dim
y_dim = mnist.train.labels.shape[1] # y data dim
h_dim = 128 # hidden layer dim
c = 0
lr = 0.0002

alpha = 0.01
gamma = 0.0001
beta = 0.001
z_lr = 0.005
J1_ = 0.01

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)


# =============================== Q(z|X) ======================================
# Gaussian MLP as encoder

Wxh = xavier_init(size=[X_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)

Whz_mu = xavier_init(size=[h_dim, Z_dim])
bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True)

Whz_var = xavier_init(size=[h_dim, Z_dim])
bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True)


def Q(X):
    h = nn.relu(X @ Wxh + bxh.repeat(X.size(0), 1))
    z_mu = h @ Whz_mu + bhz_mu.repeat(h.size(0), 1)
    z_var = h @ Whz_var + bhz_var.repeat(h.size(0), 1) # log(sigma^2)
    return z_mu, z_var

# sample z (reparameter)
def sample_z(mu, log_var):
    eps = Variable(torch.randn(mb_size, Z_dim))
    return mu + torch.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================
# Bernoulli MLP as decoder

Wzh = xavier_init(size=[Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)


def P(z):
    h = nn.relu(z @ Wzh + bzh.repeat(z.size(0), 1))
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    return X


# =============================== f1(y|X) =====================================
# MLP as Classifier f1

Wxh_f1 = xavier_init(size=[X_dim, h_dim])
bxh_f1 = Variable(torch.zeros(h_dim), requires_grad=True)

Why_f1 = xavier_init(size=[h_dim, y_dim])
bhy_f1 = Variable(torch.zeros(y_dim), requires_grad=True)


def f1(X):
    h = nn.relu(X @ Wxh_f1 + bxh_f1.repeat(X.size(0), 1))
    y = nn.sigmoid(h @ Why_f1 + bhy_f1.repeat(h.size(0), 1))
    return y


# =============================== f2(y|z) =====================================
# MLP as Classifier f2

Wzh_f2 = xavier_init(size=[Z_dim, h_dim])
bzh_f2 = Variable(torch.zeros(h_dim), requires_grad=True)

Why_f2 = xavier_init(size=[h_dim, y_dim])
bhy_f2 = Variable(torch.zeros(y_dim), requires_grad=True)


def f2(z):
    h = nn.relu(z @ Wzh_f2 + bzh_f2.repeat(z.size(0), 1))
    y = nn.sigmoid(h @ Why_f2 + bhy_f2.repeat(h.size(0), 1))
    return y


# =============================== D(dis|z) ====================================
# MLP as Discriminator

Wzh_D = xavier_init(size=[Z_dim, h_dim])
bzh_D = Variable(torch.zeros(h_dim), requires_grad=True)

Wh1 = xavier_init(size=[h_dim, 1])
bh1 = Variable(torch.zeros(1), requires_grad=True)


def D(z):
    h = nn.relu(z @ Wzh_D + bzh_D.repeat(z.size(0), 1))
    dis = nn.sigmoid(h @ Wh1 + bh1.repeat(h.size(0), 1))
    return dis


# ================================ MAIN =======================================

if len(sys.argv) < 2:
    print('Usage: python svae_mnist.py train')
    print('       python svae_mnist.py generate digit_type')
    exit(0)

stage1_params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var,
            Wzh, bzh, Whx, bhx, Wzh_f2, bzh_f2, Why_f2, bhy_f2]

stage2_params = [Wzh_D, bzh_D, Wh1, bh1]

f1_params = [Wxh_f1, bxh_f1, Why_f1, bhy_f1]

if sys.argv[1] == 'train':
# =============================== TRAINING ====================================

    stage1_solver = optim.Adam(stage1_params, lr=lr)
    stage2_solver = optim.Adam(stage2_params, lr=lr)
    f1_solver = optim.Adam(f1_params, lr=lr)


# =============================== Stage1 TRAINING =============================

    for it in range(100000):
        X, y = mnist.train.next_batch(mb_size)
        X = Variable(torch.from_numpy(X))
        y = Variable(torch.from_numpy(y)).float()

        # Forward
        z_mu, z_var = Q(X)
        z = sample_z(z_mu, z_var)
        X_sample = P(z)
        y_ = f2(z)

        # Loss
        classify_loss = nn.binary_cross_entropy(y_, y, size_average=False) / mb_size
        recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False) / mb_size
        kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var, 1))
        loss = classify_loss + recon_loss + kl_loss

        # Backward
        loss.backward()

        # Update
        stage1_solver.step()

        # Housekeeping
        for p in stage1_params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_()) # Grad set 0

        # Print and plot every now and then
        if it % 1000 == 0:
            print('Iter-{}; Stage1 Loss: {:.4}'.format(it, loss.item()))

            samples = P(z).data.numpy()[:16]

            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for i, sample in enumerate(samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

            if not os.path.exists('out/'):
                os.makedirs('out/')

            if not os.path.exists('out/mnist/'):
                os.makedirs('out/mnist/')

            plt.savefig('out/mnist/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
            c += 1
            plt.close(fig)

    print('-----------------------------------------------------------------')


# =============================== Stage2 TRAINING =============================

    for it in range(100000):
        X, _ = mnist.train.next_batch(mb_size)
        X = Variable(torch.from_numpy(X))

        # Forward
        z_mu, z_var = Q(X)
        z_real = sample_z(z_mu, z_var)

        z_fake = Variable(torch.randn(mb_size, Z_dim))

        D_real = D(z_real)
        D_fake = D(z_fake)

        # Loss
        loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake)) 

        # Backward
        loss.backward()

        # Update
        stage2_solver.step()

        # Housekeeping
        for p in stage1_params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_()) # Grad set 0

        for p in stage2_params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_()) # Grad set 0

        # Print and plot every now and then
        if it % 1000 == 0:
            print('Iter-{}; Stage2 Loss: {:.4}'.format(it, loss.item()))

    print('-----------------------------------------------------------------')

    svae_dic = {'Wxh': Wxh, 'bxh': bxh, 'Whz_mu': Whz_mu, 'bhz_mu': bhz_mu,
            'Whz_var': Whz_var, 'bhz_var': bhz_var,
            'Wzh': Wzh, 'bzh': bzh, 'Whx': Whx, 'bhx': bhx, 
            'Wzh_f2': Wzh_f2, 'bzh_f2': bzh_f2, 'Why_f2': Why_f2, 'bhy_f2': bhy_f2,
            'Wzh_D': Wzh_D, 'bzh_D': bzh_D, 'Wh1': Wh1, 'bh1': bh1}
    torch.save(svae_dic, 'out/mnist/svae.pth')


# =============================== Classifier f1 TRAINING ======================

    for it in range(100000):
        X, y = mnist.train.next_batch(mb_size)
        X = Variable(torch.from_numpy(X))
        y = Variable(torch.from_numpy(y)).float()

        # Forward
        z_mu, z_var = Q(X)
        z_ = sample_z(z_mu, z_var)
        X_ = P(z_)
        y_ = f1(X_)

        # Loss
        loss = nn.binary_cross_entropy(y_, y, size_average=False) / mb_size

        # Backward
        loss.backward()

        # Update
        f1_solver.step()

        # Housekeeping
        for p in stage1_params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_()) # Grad set 0

        for p in f1_params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_()) # Grad set 0

        # Print and plot every now and then
        if it % 1000 == 0:
            print('Iter-{}; Classifier f1 Loss: {:.4}'.format(it, loss.item()))

    print('-----------------------------------------------------------------')

    f1_dic = {'Wxh_f1': Wxh_f1, 'bxh_f1': bxh_f1, 'Why_f1': Why_f1, 'bhy_f1': bhy_f1}
    torch.save(f1_dic, 'out/mnist/f1.pth')

elif sys.argv[1] == 'generate':
    y_ = torch.zeros(mb_size, y_dim, dtype=torch.float)

    if len(sys.argv) < 3:   
        index = torch.tensor([0])
        y_.index_fill_(1, index, 1)
    else:
        index = torch.tensor([int(sys.argv[2])])
        y_.index_fill_(1, index, 1)

    svae_net = torch.load('out/mnist/svae.pth')
    f1_net = torch.load('out/mnist/f1.pth')

    Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var = svae_net['Wxh'], svae_net['bxh'], svae_net['Whz_mu'], svae_net['bhz_mu'], svae_net['Whz_var'], svae_net['bhz_var']
    Wzh, bzh, Whx, bhx =  svae_net['Wzh'], svae_net['bzh'], svae_net['Whx'], svae_net['bhx']
    Wzh_f2, bzh_f2, Why_f2, bhy_f2 = svae_net['Wzh_f2'], svae_net['bzh_f2'], svae_net['Why_f2'], svae_net['bhy_f2']
    Wzh_D, bzh_D, Wh1, bh1 = svae_net['Wzh_D'], svae_net['bzh_D'], svae_net['Wh1'], svae_net['bh1']
    Wxh_f1, bxh_f1, Why_f1, bhy_f1 = f1_net['Wxh_f1'], f1_net['bxh_f1'], f1_net['Why_f1'], f1_net['bhy_f1']

    X, y = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X))
    y = Variable(torch.from_numpy(y)).float()

    _, pred = torch.max(y, 1)
    print('Label Y: ', pred[:16])

    # Forward
    z_, _ = Q(X)
    z_ = Variable(z_, requires_grad=True)
    y2 = f2(z_)
    D_ = D(z_)
    X_ = P(z_)
    y1 = f1(X_)

    k = 0
    z_solver = optim.Adam([z_], lr=z_lr)

    for it in range(25000):
        # Loss
        J2 = nn.binary_cross_entropy(y2, y_, size_average=False) / mb_size
        J_IT = J2 + alpha * torch.mean(1 - D_) + gamma * torch.mean(torch.norm(z_, dim=1))
        J1 = nn.binary_cross_entropy(y1, y, size_average=False) / mb_size
        J_SA = J_IT + k * J1

        # Backward
        J_SA.backward(retain_graph=True)

        # Update
        z_solver.step()

        # Housekeeping
        for p in stage1_params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_()) # Grad set 0

        for p in stage2_params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_()) # Grad set 0

        for p in f1_params:
            if p.grad is not None:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_()) # Grad set 0
        
        if z_.grad is not None:
            data = z_.grad.data
            z_.grad = Variable(data.new().resize_as_(data).zero_()) # Grad set 0

        y2 = f2(z_)
        D_ = D(z_)
        X_ = P(z_)
        y1 = f1(X_)

        J1 = nn.binary_cross_entropy(y1, y, size_average=False) / mb_size
        J2 = nn.binary_cross_entropy(y2, y_, size_average=False) / mb_size
        k += z_lr * (beta * J1 - J2 + max(J1 - J1_, 0))
        k = max(0, min(k, 0.005))

        # Print and plot every now and then
        if it % 1000 == 0:
            print('Iter-{}; J_SA Loss: {:.4}'.format(it, J_SA.item()))

            _, f1_pred = torch.max(y1, 1)
            _, f2_pred = torch.max(y2, 1)
            print('f1 pred: ', f1_pred[:16])
            print('f2 pred: ', f2_pred[:16])

            attack_samples = P(z_).data.numpy()[:16]

            fig = plt.figure(figsize=(4, 4))
            gs = gridspec.GridSpec(4, 4)
            gs.update(wspace=0.05, hspace=0.05)

            for i, attack_sample in enumerate(attack_samples):
                ax = plt.subplot(gs[i])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                plt.imshow(attack_sample.reshape(28, 28), cmap='Greys_r')

            if not os.path.exists('attack/'):
                os.makedirs('attack/')

            if not os.path.exists('attack/mnist/'):
                os.makedirs('attack/mnist/')

            plt.savefig('attack/mnist/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
            c += 1
            plt.close(fig)
    
else:
    print('Usage: python svae_mnist.py train')
    print('       python svae_mnist.py generate digit_type(default = 0)')
    exit(0)

