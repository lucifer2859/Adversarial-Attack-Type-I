import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.insert(0, "/home/dchen/SVAE/classification-cifar10-pytorch/")

from models import *
from utils import progress_bar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# net =  MobileNetV2()
net = DPN92()
net_name = net.name
save_path = 'out/cifar/f1_{0}.pth'.format(net.name)
net = net.to(device)

checkpoint = torch.load(save_path)
net.load_state_dict(checkpoint['net'])

def test():
    net.eval()

    with torch.no_grad():
        inputs = torch.randn(64, 3, 32, 32).to(device)
        outputs = net(inputs)

        class_prob = torch.mean(F.softmax(outputs, dim=1), dim=0).data.cpu().numpy()

        for i in range(len(classes)):
            print('%s: %.6f' % (classes[i], class_prob[i]))

### Attack Samples Test ###

### Validation Dataset Test ###
test()