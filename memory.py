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
    """Memory bank for NTM."""
    pi_2 = 3.14159/2
    def __init__(self, N=2000, C=512, H=7, W=7):
        """Initialize the Memory matrix.
        N: Number of cubes in the memory.
        C: Channel of each cube in the memory
        H: Height of each cube in the memory
        W: Width of each cube in the memory
        """
        super(Memory, self).__init__()
        self.N, self.C, self.H, self.W = N, C, H, W
        self.register_buffer('memory', torch.zeros(N, C, H, W))
        nn.init.uniform_(self.memory)
        self._normalize_memory()

    def size(self):
        return self.memory.size()

    def read(self, key):
        w = self._address(key)
        return torch.sum(w * self.memory, dim=1)

    def write(self, key):
        w = self._address(key)
        memory = ((1 - w) * self.memory.data).sum(dim=0)
        knowledge = (w * key.unsqueeze(1)).sum(dim=0)
        self.memory.data = memory + knowledge
        self._normalize_memory()

    def _address(self, key):
        key = self._normalize(key)
        key = key.view(key.size(0), 1, -1)
        memory = self.memory.view(self.N, -1)
        w = F.softmax((F.cosine_similarity(memory, key, dim=-1)*self.pi_2).tan(), dim=1)
        return w.view(-1, self.N, 1, 1, 1)

    def _normalize_memory(self):
        self.memory.data /= self.memory.sum(dim=[1,2,3], keepdim=True) + 1e-7

    def _normalize(self, key):
        return key/(key.sum(dim=[1,2,3],keepdim=True) + 1e-7)


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    logger =  SummaryWriter('runs/test3')

    N, B, C, H, W = 5, 5, 1, 3, 3
    mem = Memory(N, C, H, W)

    key = torch.rand(B, C, H, W)
    key /= (key.sum(dim=[1,2,3],keepdim=True)+1e-7)

    say = torch.rand(B, C, H, W)
    say /= (say.sum(dim=[1,2,3],keepdim=True)+1e-7)

    logger.add_images('key1', key/key.max())
    logger.add_images('say1', say/say.max())

    for i in range(10):
        mem.write(key)

    rkey = mem.read(key) 
    logger.add_images('key2', rkey/rkey.max())

    for i in range(3):
        mem.write(say)
    
    rkey = mem.read(key)
    rsay = mem.read(say)

    logger.add_images('key3', rkey/rkey.max())
    logger.add_images('say2', rsay/rsay.max())
