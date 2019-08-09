import os
import copy
import tqdm
import torch
import os.path
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision.models.vgg import VGG
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection

from coder import Encoder, Decoder # for coco
# from coders import Encoder, Decoder # for mnist
from memory import Memory
from head import ReadHead, WriteHead


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        coding = self.encoder(x)
        output = self.decoder(coding)
        return output


class Interestingness(nn.Module):
    def __init__(self, autoencoder, N, C, H, W):
        super().__init__()
        self.ae = autoencoder
        self._freezing_autoencoder()
        self.memory = Memory(N, C, H, W)
        self.reader = ReadHead(self.memory)
        self.writer = WriteHead(self.memory)

    def forward(self, x):
        coding = self.ae.encoder(self.ae, x)
        states = self.reader(coding)
        self.writer(coding) 
        output = self.ae.decoder(self.ae, states)
        return output

    def _freezing_autoencoder(self):
        for param in self.ae.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    ## for mnist data
    # x = torch.rand(15, 1, 28, 28)
    # ae = AutoEncoder()
    # net = Interestingness(ae, 200, 6, 4, 4)
    # y = net(x)

    ## for coco data
    x = torch.rand(15, 3, 224, 224)
    ae = AutoEncoder()
    net = Interestingness(ae, 2000, 512, 7, 7)
    y = net(x)