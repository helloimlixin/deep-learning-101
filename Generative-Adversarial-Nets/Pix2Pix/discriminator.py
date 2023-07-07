#
# Created on Fri Jul 07 2023
#
# Copyright (c) 2023 Xin Li
# Email: helloimlixin@gmail.com
# All rights reserved.
#
# This file is part of Pix2Pix.

import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode='reflect'), # the reflect padding can help to avoid the edge effect
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.conv(x)

# x, y <- concatenate along the channel dimension
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]): # 256 -> 30x30
        super().__init__()
        self.initial = nn.Sequential(
            # here in_channels*2 because we concatenate the input image and the target image
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for feature in features[1:]: # skip the initial block
            layers.append(CNNBlock(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature
        
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))

        self.model = nn.Sequential(*layers)
    
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1) # concatenate along the channel dimension
        x = self.initial(x)
        
        return self.model(x)

'''
Some basic tests that should always be there when creating a new model
'''
def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=3)
    predictions = model(x, y)
    print(predictions.shape)

if __name__ == "__main__":
    test()