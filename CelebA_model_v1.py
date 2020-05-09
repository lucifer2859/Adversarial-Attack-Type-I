import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-6

class Encoder(nn.Module):
    def __init__(self, dim_z):
        super(Encoder, self).__init__()

        self.dim_z = dim_z

        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 5, 1, 2)
        self.conv4 = nn.Conv2d(64, 128, 5, 2, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 5, 1, 2)
        self.conv6 = nn.Conv2d(128, 256, 5, 2, 2)
        self.bn3 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 8 * 8, 2048)
        self.fc2 = nn.Linear(2048, dim_z * 2)

    def forward(self, x):
        dim_z = self.dim_z

        x = F.elu(self.bn1(self.conv2(F.elu(self.conv1(x)))))
        x = F.elu(self.bn2(self.conv4(F.elu(self.conv3(x)))))
        x = F.elu(self.bn3(self.conv6(F.elu(self.conv5(x)))))
        x = x.view(x.size()[0], -1)
        x = self.fc2(torch.tanh(self.fc1(x)))

        mu = x[:, :dim_z]
        sigma = EPSILON + F.softplus(x[:, dim_z:])
        return mu, sigma

class Generator(nn.Module):
    def __init__(self, dim_z):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(dim_z, 2048)
        self.fc2 = nn.Linear(2048, 256 * 8 * 8)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(256, 128, 5, 1, 2)
        self.conv2 = nn.Conv2d(128, 128, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, 5, 1, 2)
        self.conv4 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, 5, 1, 2)
        self.conv6 = nn.Conv2d(32, 3, 5, 1, 2)


    def forward(self, z):
        x = self.fc2(F.elu(self.fc1(z)))
        x = x.view(-1, 256, 8, 8)
        x = F.elu(self.bn1(x))
        x = F.interpolate(x, size=(16, 16), mode='bilinear', align_corners=True)
        x = F.elu(self.bn2(self.conv2(F.elu(self.conv1(x)))))
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=True)
        x = F.elu(self.bn3(self.conv4(F.elu(self.conv3(x)))))
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
        x = torch.sigmoid(self.conv6(F.elu(self.conv5(x))))

        return x

class Classifier(nn.Module):
    def __init__(self, dim_z):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(dim_z, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, z):
        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, dim_z):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(dim_z, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, z):
        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        return x


def build_CelebA_Model(dim_z=1024):
    return Encoder(dim_z), Generator(dim_z), Classifier(dim_z), Discriminator(dim_z)
