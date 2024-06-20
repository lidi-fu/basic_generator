import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGenerator(nn.Module):
    def __init__(self, noise_dim=100):
        super(ConvGenerator, self).__init__()
        self.fc = nn.Linear(noise_dim, 256*8*8)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.tanh(self.deconv4(x))
        return x

# To be saved as models/generator.py
