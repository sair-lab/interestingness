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


class HeadBase(nn.Module):
    def __init__(self, memory):
        super(HeadBase, self).__init__()
        self.memory = memory
        _,self.C,self.H,self.W = memory.size()
        self.embeddings_size = self.C*self.H*self.W


class ReadHead(HeadBase):
    def __init__(self, memory):
        super(ReadHead, self).__init__(memory)
        self.transform = nn.Conv2d(in_channels=self.C, out_channels=self.C, kernel_size=1, groups=self.C)

    def forward(self, embeddings):
        # embeddings = self.transform(embeddings)
        return self.memory.read(embeddings)


class WriteHead(HeadBase):
    def __init__(self, memory):
        super(WriteHead, self).__init__(memory)
        self.transform = nn.Conv2d(in_channels=self.C, out_channels=self.C, kernel_size=1, groups=self.C)

    def forward(self, embeddings):
        # embeddings = self.transform(embeddings)
        self.memory.write(embeddings)


if __name__ == "__main__":
    from memory import Memory
    import torchvision
    from torch.utils.tensorboard import SummaryWriter
    logger =  SummaryWriter('runs/head')
    N, B, C, H, W = 5, 1, 3, 2, 2

    memory = Memory(N, C, H, W)
    writer = WriteHead(memory)
    reader = ReadHead(memory)

    coding = torch.rand(B, C, H, W)
    logger.add_images('coding', coding)
    logger.add_images('memory', memory.memory/memory.memory.max())

    for i in range(100):
        writer(coding)
        x = reader(coding)
        logger.add_images('memory', memory.memory/memory.memory.max())
