#!/usr/bin/env python3
 
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchutil import CorrelationSimilarity, rolls2d


class Memory(nn.Module):
    pi_2 = 3.14159/2
    def __init__(self, N=2000, C=512, H=7, W=7, rr=1, wr=1):
        """Initialize the Memory.
        N: Number of cubes in the memory.
        C: Channel of each cube in the memory
        H: Height of each cube in the memory
        W: Width of each cube in the memory
        rr: reading rate [rr > 0] ($\gamma_r$ in the paper)
        wr: writing rate [wr > 0] ($\gamma_w$ in the paper)
        """
        super(Memory, self).__init__()
        self.N, self.C, self.H, self.W = N, C, H, W
        self.set_learning_rate(rr, wr)
        self.register_buffer('memory', torch.zeros(N, C, H, W))
        self.register_buffer('usage', torch.zeros(1, N))
        nn.init.kaiming_uniform_(self.memory)
        self._normalize_memory()
        self.similarity = CorrelationSimilarity((H,W))

    def set_learning_rate(self, rr, wr):
        self.rr, self.wr = rr, wr

    def size(self):
        return self.memory.size()

    def read(self, key):
        key = self._normalize(key)
        w, trans = self._correlation_address(key)
        memory = rolls2d(self.memory, -trans)
        return (w * memory).sum(dim=1)

    def write(self, keys):
        for i in range(keys.size(0)):
            key = self._normalize(keys[i]).unsqueeze(0)
            w = self._address(key)
            memory = (1 - w) * self.memory.data
            knowledge = w * key.unsqueeze(1)
            self.memory.data = (memory + knowledge).mean(0)
            self._normalize_memory()

    def _correlation_address(self, key):
        # self.rw: reading weights, saved for human interaction package
        # Go https://github.com/wang-chen/interaction.git
        self.rw, trans = self.similarity(key, self.memory)
        w = F.softmax((self.rw*self.pi_2).tan()*self.rr, dim=1)
        return w.view(-1, self.N, 1, 1, 1), trans

    def _address(self, key):
        key = key.view(key.size(0), 1, -1)
        memory = self.memory.view(self.N, -1)
        w = F.cosine_similarity(memory, key, dim=-1)
        w[w<self.usage] = ((w+1)*(1-self.usage)-1)[w<self.usage]
        w = F.softmax((w*self.pi_2).tan()*self.wr, dim=1)
        self.usage = torch.max((1-w)*self.usage, w*(1-self.usage))
        return w.view(-1, self.N, 1, 1, 1)

    def _normalize_memory(self):
        self.memory.data = self._normalize(self.memory.data)

    def _normalize(self, x):
        return x


if __name__ == "__main__":
    '''
    Memory Tests
    '''
    from torch.utils.tensorboard import SummaryWriter
    import time
    logger =  SummaryWriter('runs/memory-'+str(time.time()))

    N, B, C, H, W = 10, 1, 1, 3, 3
    mem = Memory(N, C, H, W)

    def criterion(a, b):
        a = a.view(-1)
        b = b.view(-1)
        return F.cosine_similarity(a, b, dim = 0)

    def _normalize(x):
        return x

    def add_loss(i):
        r1 = mem.read(k1)
        r2 = mem.read(k2)
        loss1 = criterion(k1, r1)
        loss2 = criterion(k2, r2)
        logger.add_scalars('Loss', {'k1': loss1, 'k2': loss2}, i)
        logger.add_images('k1r', r1, i)
        logger.add_images('k2r', r2, i)
        logger.add_images('memory', mem.memory.data, i)

    k1 = _normalize(torch.randn(B, C, H, W))
    k2 = _normalize(torch.randn(B, C, H, W))

    logger.add_images('k1', k1/k1.max())
    logger.add_images('k2', k2/k2.max())

    for i in range(5):
        add_loss(i)
        mem.write(k1)

    for i in range(5, 18):
        add_loss(i)
        mem.write(k2)

    for i in range(18, 25):
        add_loss(i)
        mem.write(k1)

    f1 = mem.read(k1)
    f2 = mem.read(k2)
