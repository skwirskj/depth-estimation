import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import sqrt

original_model = models.densenet169(pretrained=True)

class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        
        # Encoder
        self.features = nn.Sequential(
            *list(original_model.features.children())[:-1] # Exclude classifier layer
        )

        # Decoder
        self.decoder = nn.Sequential(
            # TODO define decoding layers from paper
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=1664, out_channels=832, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(832, 832, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode= 'bilinear'),
            nn.Conv2d(832, 416, 3, 1, 1),
            nn.Conv2d(416, 416, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(416, 208, 3, 1, 1),
            nn.Conv2d(208, 208, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(208, 104, 3, 1, 1),
            nn.Conv2d(104, 104, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(104, 1, 3, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        print(x.shape)
        x = self.decoder(x)
        print(x.shape)
        return x