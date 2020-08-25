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
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.vgg import VGG
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection

from memory import Memory
from torchutil import Split2d, Merge2d


class Interestingness(nn.Module):
    def __init__(self, autoencoder, N, C, H, W, h, w):
        super().__init__()
        self.ae = autoencoder
        self.memory = Memory(N, C, h, w)
        self.split2d = Split2d(kernel_size=(h, w))
        self.merge2d = Merge2d(output_size=(H, W), kernel_size=(h, w))
        self.set_parameters()
        self.set_train(False)

    def forward(self, x):
        coding = self.ae.encoder(x)
        coding = self.split2d(coding)
        if self.train:
            self.memory.write(coding)
            states = self.memory.read(coding)
            states = self.merge2d(states)
            output = self.ae.decoder(states)
            return output
        else:
            # self.coding, self.states, saved for human interaction package
            # Go https://github.com/wang-chen/interaction.git
            self.states, self.coding = self.memory.read(coding), coding
            self.memory.write(coding)
            self.reads = self.merge2d(self.states)
            return 1-F.cosine_similarity(coding.view(coding.size(1),-1), self.reads.view(self.reads.size(1),-1),dim=-1).mean()

    def output(self):
        return self.ae.decoder(self.reads)

    def listen(self, x):
        coding = self.ae.encoder(x)
        coding = self.split2d(coding)
        states = self.memory.read(coding)
        states = self.merge2d(states)
        return self.ae.decoder(states)

    def set_parameters(self):
        for param in self.ae.parameters():
            param.requires_grad = False
        for param in self.memory.parameters():
            param.requires_grad = True

    def set_train(self, train):
        self.train = train


if __name__ == "__main__":
    from autoencoder import AutoEncoder
    x = torch.rand(15, 3, 320, 320)
    ae = AutoEncoder()
    net = Interestingness(ae, 200, 512, 10, 10, 10, 10)
    y = net(x)
