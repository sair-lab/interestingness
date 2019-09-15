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
 
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class Memory(nn.Module):
    pi_2 = 3.14159/2
    def __init__(self, N=2000, C=512, H=7, W=7):
        """Initialize the Memory Tensors.
        N: Number of cubes in the memory.
        C: Channel of each cube in the memory
        H: Height of each cube in the memory
        W: Width of each cube in the memory
        """
        super(Memory, self).__init__()
        self.N, self.C, self.H, self.W = N, C, H, W
        self.register_buffer('memory', torch.zeros(N, C, H, W))
        nn.init.kaiming_uniform_(self.memory)
        self._normalize_memory()

    def size(self):
        return self.memory.size()

    def read(self, key):
        key = self._normalize(key)
        w = self._address(key)
        return torch.sum(w * self.memory, dim=1)

    def write(self, key):
        key = self._normalize(key)
        w = self._address(key)
        memory = ((1 - w) * self.memory.data).mean(dim=0)
        knowledge = (w * key.unsqueeze(1)).mean(dim=0)
        self.memory.data = memory + knowledge
        self._normalize_memory()

    def _address(self, key):
        key = key.view(key.size(0), 1, -1)
        memory = self.memory.view(self.N, -1)
        w = F.softmax((F.cosine_similarity(memory, key, dim=-1)*self.pi_2).tan(), dim=1)
        return w.view(-1, self.N, 1, 1, 1)

    def _normalize_memory(self):
        self.memory.data = self._normalize(self.memory.data)

    def _normalize(self, x):
        return x


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    import time
    logger =  SummaryWriter('runs/memory-'+str(time.time()))

    N, B, C, H, W = 1, 1, 1, 3, 3
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
