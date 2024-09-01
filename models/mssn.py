import torch
import torch.nn as nn

class MSSN(nn.Module):
    def __init__(self, channels=16, num_dense_blocks=5):
        super(MSSN, self).__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            DenseBlock(channels, num_dense_blocks)
        )
        self.downsample = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose2d(channels * 2, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.downsample(x)
        x = self.upsample(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
