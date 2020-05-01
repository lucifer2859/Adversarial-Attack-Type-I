import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import sys
sys.path.insert(0, "/home/dchen/SVAE/classification-cifar10-pytorch/")

from models import *
from utils import progress_bar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='/home/dchen/dataset', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# net =  MobileNetV2()
net = DPN92()
net_name = net.name
save_path = 'out/cifar/f1_{0}.pth'.format(net.name)
net = net.to(device)

checkpoint = torch.load(save_path)
net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()

def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d / %d)'
                % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))    

### Attack Samples Test ###

### Validation Dataset Test ###
test()