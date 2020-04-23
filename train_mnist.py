import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

import mnist_data

np.random.seed(996)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

EPSILON = 1e-6

IMAGE_SIZE = 28

epoch_num = 600
batch_size = 64
classes = 10
lr = 0.00035

if not os.path.exists('out/'):
    os.makedirs('out/')

if not os.path.exists('out/mnist/'):
    os.makedirs('out/mnist/')

if not os.path.exists('attack/'):
    os.makedirs('attack/')

if not os.path.exists('attack/mnist/'):
    os.makedirs('attack/mnist/')

# class MLP(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
        
#         self.fc1 = nn.Linear(28 * 28, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = x.view(x.size()[0], -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Linear(84, 15)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# f1 = MLP().to(device)
f1 = CNN().to(device)

# =============================== TRAINING ====================================

f1_solver = optim.Adam(f1.parameters(), lr=lr)

best_CLA_loss = 1000.

best_epoch = -1

x_train, y_train, x_test, y_test = mnist_data.load_mnist(reshape=True, twoclass=None, binary=True, onehot=False)

n_train_samples = x_train.shape[0]
n_test_samples = x_test.shape[0]
total_train_batch = n_train_samples // batch_size
total_test_batch = n_test_samples // batch_size
            
for epoch in range(0, epoch_num):
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

        f1_solver.zero_grad()

        # Forward
        y_hat = f1(X)

        # Loss
        loss = nn.CrossEntropyLoss()(y_hat, y)       

        # Backward
        loss.backward()

        # Update
        f1_solver.step()

    with torch.no_grad():
        # validate after each epoch
        val_loss, val_acc = 0., 0.

        for i in range(total_test_batch):
            offset = (i * batch_size) % n_test_samples
            batch_xs = x_test[offset:(offset + batch_size), ...]
            batch_ys = y_test[offset:(offset + batch_size), ...]

            X, y = torch.from_numpy(batch_xs).float().to(device), torch.from_numpy(batch_ys).long().to(device)

            y_hat = f1(X)

            # Loss
            classify_loss = nn.CrossEntropyLoss()(y_hat, y)
            _, predicted = torch.max(y_hat.data, 1)
            acc = torch.mean(torch.eq(predicted, y).float())

            val_loss += loss.item()
            val_acc += acc.item()

        print('Epoch %d, val_loss: %f' % (epoch, val_loss / total_test_batch))
            
        if (val_loss / total_test_batch) < best_CLA_loss:
            torch.save(f1.state_dict(), 'out/mnist/f1_best_%d.pth' % (epoch))
            best_CLA_loss = val_loss / total_test_batch

            if best_epoch >= 0:
                os.system('rm out/mnist/f1_best_%d.pth' % (best_epoch))

            best_epoch = epoch