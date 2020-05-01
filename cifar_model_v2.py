import torch
import torch.nn as nn
import torch.nn.functional as F
import baseNet

EPSILON = 1e-6

class Encoder(nn.Module):
    def __init__(self, dim_z, device):
        super(Encoder, self).__init__()

        self.dim_z = dim_z

        self.conv_channels_up = nn.Conv2d(3, 256, 1)
        convList = []
        convList.append(baseNet.SampleNet(downSample=True, in_channels=256, out_channels=128, device=device))
        convList.append(baseNet.SampleNet(downSample=True, in_channels=128, out_channels=64, device=device))

        self.convList = nn.Sequential(*convList)

        self.fc1 = nn.Linear(64 * 8 * 8, 2048)
        self.fc2 = nn.Linear(2048, dim_z * 2)

    def forward(self, x):
        dim_z = self.dim_z

        x = self.convList(F.leaky_relu_(self.conv_channels_up(x)))
        x = x.view(x.size()[0], -1)
        x = self.fc2(torch.tanh(self.fc1(x)))

        mu = x[:, :dim_z]
        sigma = EPSILON + F.softplus(x[:, dim_z:])
        return mu, sigma

class Generator(nn.Module):
    def __init__(self, dim_z, device):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(dim_z, 2048)
        self.fc2 = nn.Linear(2048, 64 * 8 * 8)
        self.tconv_channels_down = nn.ConvTranspose2d(256, 3, 1)
        convList = []
        convList.append(baseNet.SampleNet(downSample=False, in_channels=64, out_channels=128, device=device))
        convList.append(baseNet.SampleNet(downSample=False, in_channels=128, out_channels=256, device=device))

        self.convList = nn.Sequential(*convList)

    def forward(self, z):
        x = self.fc2(F.elu(self.fc1(z)))
        x = x.view(-1, 64, 8, 8)
        x = F.leaky_relu_(self.tconv_channels_down(self.convList(x)))

        return x

class Classifier(nn.Module):
    def __init__(self, dim_z):
        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(dim_z, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

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


def build_CIFAR_Model(dim_z=1024, device=torch.device("cpu")):
    return Encoder(dim_z, device=device).to(device), Generator(dim_z, device=device).to(device), Classifier(dim_z).to(device), Discriminator(dim_z).to(device)
