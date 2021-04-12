#!/usr/bin/env python3

import os
import copy
import torch
import os.path
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.utils.data as Data


def encoder(model):
    models = {'vgg': VGG,
              'resnet': ResNet,
              'mobilenet': MobileNetV2}
    Model = models[model]
    return Model()


class AutoEncoder(nn.Module):
    def __init__(self, model='vgg'):
        super().__init__()
        self.encoder = encoder(model)
        self.decoder = Decoder()

    def forward(self, x):
        coding = self.encoder(x)
        output = self.decoder(coding)
        return output


class VGG(models.VGG):
    def __init__(self, pretrained=True, requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(models.vgg16().features)

        if pretrained:
            self.load_state_dict(models.vgg16(pretrained=True).state_dict())

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        x = self.features(x)
        return x


class ResNet(models.ResNet):
    def __init__(self, pretrained=True, requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2])

        if pretrained:
            self.load_state_dict(models.resnet18(pretrained=True).state_dict())

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:
            del self.fc

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class MobileNetV2(models.MobileNetV2):
    def __init__(self, pretrained=True, requires_grad=True, remove_fc=True, show_params=False):
        super().__init__()

        if pretrained:
            self.load_state_dict(models.mobilenet_v2(pretrained=True).state_dict())

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        return self.features(x)


class Decoder(nn.Module):
    def __init__(self, in_channels=512): # Use 1280 for MobileNetV2
        super().__init__()
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(in_channels, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        x = self.bn1(self.relu(self.deconv1(x)))  # size=(N, 512, x.H/16, x.W/16)
        x = self.bn2(self.relu(self.deconv2(x)))  # size=(N, 256, x.H/8, x.W/8)
        x = self.bn3(self.relu(self.deconv3(x)))  # size=(N, 128, x.H/4, x.W/4)
        x = self.bn4(self.relu(self.deconv4(x)))  # size=(N, 64, x.H/2, x.W/2)
        x = self.bn5(self.relu(self.deconv5(x)))  # size=(N, 32, x.H, x.W)
        x = self.classifier(x)                    # size=(N, n_class, x.H/1, x.W/1)
        return x                                  # size=(N, n_class, x.H/1, x.W/1)


if __name__ == "__main__":
    from dataset import SubTF
    from torchutil import show_batch
    import torchvision.transforms as transforms

    transform = transforms.Compose([
            transforms.CenterCrop(tuple([320, 320])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data = SubTF(root='/data/datasets', train=True, transform=transform)
    loader = Data.DataLoader(dataset=data, batch_size=1, shuffle=True)

    net, best_loss = torch.load('saves/resnet.pt')

    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = net(inputs)
            show_batch(torch.cat([inputs, outputs], dim=0), name='test', waitkey=1000)
