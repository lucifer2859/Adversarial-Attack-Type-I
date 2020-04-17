import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-6

class Encoder(nn.Module):
    def __init__(self, dim_z):
        super(Encoder, self).__init__()

        self.dim_z = dim_z

        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 2, 1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, dim_z * 2)

    def forward(self, x):
        dim_z = self.dim_z

        x = F.elu(self.conv2(F.elu(self.conv1(x))))
        x = F.elu(self.conv4(F.elu(self.conv3(x))))
        x = x.view(x.size()[0], -1)
        x = self.fc2(torch.tanh(self.fc1(x)))

        mu = x[:, :dim_z]
        sigma = EPSILON + F.softplus(x[:, dim_z:])
        return mu, sigma

class Generator(nn.Module):
    def __init__(self, dim_z):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(dim_z, 128)
        self.fc2 = nn.Linear(128, 64 * 7 * 7)
        self.conv1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, z):
        x = F.elu(self.fc2(F.elu(self.fc1(z))))
        x = x.view(-1, 64, 7, 7)
        x = F.interpolate(x, size=(14, 14), mode='bilinear', align_corners=True)
        x = F.elu(self.conv2(F.elu(self.conv1(x))))
        x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=True)
        x = torch.sigmoid(self.conv4(F.elu(self.conv3(x))))
        return x

class Classifier(nn.Module):
    def __init__(self, dim_z):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(dim_z, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, z):
        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, dim_z):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(dim_z, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, z):
        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x
        

def build_MNIST_Model(dim_z=32):
    return Encoder(dim_z), Generator(dim_z), Classifier(dim_z), Discriminator(dim_z)
