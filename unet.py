import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import sqrt

class Dense(nn.Module):
  def __init__(self):
    super(Dense, self).__init__()
    ########################################################################
    # TODO: Implement a sematic segmentation model                         #
    ########################################################################
    # Using kernel 3x3 and stride 1 and padding 1 to keep the input
    # and the output the same size. 
    self.conv1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1) # output shape of HxW
    self.relu = nn.LeakyReLU(inplace=True)
    self.conv1_2= nn.Conv2d(64, 64, 3, stride=1, padding=1) # output shape of HxW
    #nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=(2,2)) # output is now H/2xW/2
    self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1) # output is H/2xW/2
    #nn.ReLU(inplace=True)
    self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
    #nn.ReLU(inplace=True)
    #nn.MaxPool2d(kernel_size=(2,2)) # output is now H/4xW/4
    self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
    #nn.ReLU(inplace=True)
    self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
    #nn.ReLU(inplace=True)
    #nn.MaxPool2d(kernel_size=(2,2)) # output is now H/8xW/8
    self.middle = nn.Sequential(
        nn.Conv2d(256, 256, 3, stride=1, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(256, 256, 3, stride=1, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(256, 256, 3, stride=1, padding=1),
    )
    self.upsample = nn.Upsample(scale_factor=2, mode="bilinear") # output is now H/2xW/2
    self.dconv3_1 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
    #nn.ReLU(inplace=True)
    self.dconv3_2 = nn.Conv2d(256, 128, 3, padding=1, stride=1)
    #nn.ReLU(inplace=True)
    #nn.Upsample(scale_factor=2, mode="bilinear") # output is now HxW
    self.dconv2_1 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
    #nn.ReLU(inplace=True)
    self.dconv2_2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)

    self.dconv1_1 = nn.Conv2d(64, 1, 1, stride=1, padding=0)
    
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################

  def forward(self, x):
    ########################################################################
    # TODO: Implement the forward pass                                     #
    ########################################################################
    x = self.relu(self.conv1_1(x))
    conv1 = self.relu(self.conv1_2(x))
    x = self.maxpool(conv1)

    x = self.relu(self.conv2_1(x))
    conv2 = self.relu(self.conv2_2(x))
    x = self.maxpool(conv2)

    x = self.relu(self.conv3_1(x))
    conv3 = self.relu(self.conv3_2(x))
    x = self.maxpool(conv3)
    
    x = self.middle(x)

    dconv3 = self.upsample(x)
    uconv3 = torch.cat((dconv3, conv3), dim=1) # concatination layer
    x = self.relu(self.dconv3_1(uconv3))
    x = self.relu(self.dconv3_2(x))
    dconv2 = self.upsample(x)
    uconv2 = torch.cat((dconv2, conv2), dim=1)
    x = self.relu(self.dconv2_1(uconv2))
    x = self.relu(self.dconv2_2(x))
    x = self.dconv1_1(x)
  
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
    return x
