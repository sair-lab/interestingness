# Copyright <2019> <Chen Wang [https://chenwang.site], Carnegie Mellon University>

# Redistribution and use in source and binary forms, with or without modification, are 
# permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of 
# conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list 
# of conditions and the following disclaimer in the documentation and/or other materials 
# provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be 
# used to endorse or promote products derived from this software without specific prior 
# written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH 
# DAMAGE.

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
        self.memory = Memory(N, C, H, W)
        self.reader = ReadHead(self.memory)
        self.writer = WriteHead(self.memory)
        self.set_parameters()

    def forward(self, x):
        coding = self.ae.encoder(x)
        states = self.reader(coding)
        self.writer(coding)
        output = self.ae.decoder(states)
        return output

    def set_parameters(self):
        for param in self.ae.parameters():
            param.requires_grad = False
        for param in self.memory.parameters():
            param.requires_grad = False
        for param in self.reader.parameters():
            param.requires_grad = True
        for param in self.writer.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    ## for mnist data
    # x = torch.rand(15, 1, 28, 28)
    # ae = AutoEncoder()
    # net = Interestingness(ae, 200, 6, 4, 4)
    # y = net(x)

    ## for coco data
    x = torch.rand(15, 3, 224, 224)
    ae = AutoEncoder()
    net = Interestingness(ae, 200, 512, 7, 7)
    y = net(x)
