import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import mnist_data

batch_size = 64
classes = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

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
        self.fc3 = nn.Linear(84, 10)

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

f1.load_state_dict(torch.load('out/mnist/f1_best_178.pth'))

total_loss = 0.
total_acc = 0.

### Attack Samples Test ###

'''
total_ori_loss = 0.
total_ori_acc = 0.

test_path = 'attack/mnist/data_1/'
testloader = mnist_data.load_data(test_path)

for X, y in testloader:
    X = X.float().to(device)
    y = y.long().to(device)
    y_ori = (y - 1) % classes

    y_hat = f1(X)

    total_loss += nn.CrossEntropyLoss()(y_hat, y).item()
    _, predicted = torch.max(y_hat.data, 1)
    total_acc += torch.mean(torch.eq(predicted, y).float()).item()

    total_ori_loss += nn.CrossEntropyLoss()(y_hat, y_ori).item()
    _, predicted = torch.max(y_hat.data, 1)
    total_ori_acc += torch.mean(torch.eq(predicted, y_ori).float()).item()

print('Total Attack Loss: %.6f, Total Attack Acc: %f' % (total_loss / len(testloader), 
                                                            total_acc / len(testloader)))
print('Total Original Loss: %.6f, Total Original Acc: %f' % (total_ori_loss / len(testloader),
                                                                total_ori_acc / len(testloader)))
'''

### Validation Dataset Test ###

_, _, x_test, y_test = mnist_data.load_mnist(reshape=True, twoclass=None, binary=True, onehot=False)
n_test_samples = x_test.shape[0]
total_test_batch = n_test_samples // batch_size

for i in range(total_test_batch):
    offset = (i * batch_size) % n_test_samples
    batch_xs = x_test[offset:(offset + batch_size), ...]
    batch_ys = y_test[offset:(offset + batch_size), ...]

    X, y = torch.from_numpy(batch_xs).float().to(device), torch.from_numpy(batch_ys).long().to(device)

    y_hat = f1(X)

    total_loss += nn.CrossEntropyLoss()(y_hat, y).item()
    _, predicted = torch.max(y_hat.data, 1)
    total_acc += torch.mean(torch.eq(predicted, y).float()).item()

print('Val Loss: %f, Val Acc: %f' % (total_loss / total_test_batch, 
                                        total_acc / total_test_batch))