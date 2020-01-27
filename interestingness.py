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
from coder import Encoder, Decoder, LogVar
from torchutil import Split2d, Merge2d, CosineSimilarity

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        coding = self.encoder(x)
        output = self.decoder(coding)
        return self.criterion(x, output)


class VAE(AE):
    def __init__(self):
        super().__init__()
        self.logvar = LogVar()
        self.criterion = nn.MSELoss(reduction='sum')

    def forward(self, x):
        coding = self.encoder(x)
        logvar = self.logvar(coding)
        kld = self.KLD(coding, logvar)
        coding = self.reparameterize(coding, logvar)
        output = self.decoder(coding)
        mse = self.criterion(x, output)
        return  mse + kld

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def KLD(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class Interestingness(nn.Module):
    def __init__(self, autoencoder, N, C, H, W, h, w):
        super().__init__()
        self.ae = autoencoder
        self.memory = Memory(N, C, h, w)
        self.split2d = Split2d(kernel_size=(h, w))
        self.merge2d = Merge2d(output_size=(H, W), kernel_size=(h, w))
        self.similarity = CosineSimilarity()
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
            states = self.memory.read(coding)
            self.memory.write(coding)
            states = self.merge2d(states)
            output = self.ae.decoder(states)
            return output, 1-F.cosine_similarity(coding.view(coding.size(1),-1), states.view(states.size(1),-1),dim=-1).mean()

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


AutoEncoder = AE

if __name__ == "__main__":
    ## for coco data
    x = torch.rand(15, 3, 384, 384)
    ae = AE()
    ae = VAE()
    net = Interestingness(ae, 200, 512, 12, 12, 12, 12)
    y = net(x)
