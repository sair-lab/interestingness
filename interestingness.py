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

from coder import Encoder, Decoder
from memory import Memory
from head import ReadHead, WriteHead

class Interest(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        coding = self.encoder(x)
        output = self.decoder(coding)
        return output


class Interestingness(nn.Module):
    def __init__(self, N, C, H, W):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.memory = Memory(N, C, H, W)
        self.reader = ReadHead(self.memory)
        self.writer = WriteHead(self.memory)

    
    def forward(self, x):
        coding = self.encoder(x)
        states = self.reader(coding)
        self.writer(coding) 
        output = self.decoder(states)
        return output

if __name__ == "__main__":
    x = torch.rand(15, 3, 224, 224)
    net = Interestingness(2000, 512, 7, 7)
    y = net(x)