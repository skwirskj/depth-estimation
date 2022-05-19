import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import sqrt

original_model = models.resnet18(pretrained=True)

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        
        # Encoder
        self.features = nn.Sequential(
            *list(original_model.children())[:-3] # Exclude classifier layer (and another conv block)
        )

        # Decoder
        self.decoder = nn.Sequential(
            # TODO define decoding layers from paper
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode= 'bilinear'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 1, 1, 1, 0)

            #nn.Upsample(scale_factor=2, mode='bilinear'),
            #nn.Conv2d(64, 32, 3, 1, 1),
            #nn.Conv2d(32, 32, 3, 1, 1),
            #nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #nn.Conv2d(32, 1, 1, 1, 0)
        )
        '''
        # OLD DECODER
        self.decoder = nn.Sequential(
            # TODO define decoding layers from paper
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode= 'bilinear'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 1, 1, 1, 0)
        )
        '''
    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = self.decoder(x)
        #print(x.shape)
        return x