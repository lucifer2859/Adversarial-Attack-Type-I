import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import mnist_data

batch_size = 64
classes = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

f1 = Classifier().to(device)
f1.load_state_dict(torch.load('out/mnist/f1_best_178.pth'))

test_path = 'attack/mnist/data_1/'
testloader = mnist_data.load_data(test_path)

total_loss = 0.
total_acc = 0.
total_ori_loss = 0.
total_ori_acc = 0.

for X, y in testloader:
    X = X.float().to(device)
    y = y.long().to(device)
    y_ori = (y - 1) % classes

    y_hat = f1(X)

    total_loss += nn.CrossEntropyLoss()(y_hat, y)
    _, predicted = torch.max(y_hat.data, 1)
    total_acc += torch.mean(torch.eq(predicted, y).float())

    total_ori_loss += nn.CrossEntropyLoss()(y_hat, y_ori)
    _, predicted = torch.max(y_hat.data, 1)
    total_ori_acc += torch.mean(torch.eq(predicted, y_ori).float())

print('Total Attack Loss: %.6f, Total Attack Acc: %f' % (total_loss / len(testloader), 
                                                            total_acc / len(testloader)))
print('Total Original Loss: %.6f, Total Original Acc: %f' % (total_ori_loss / len(testloader),
                                                                total_ori_acc / len(testloader)))