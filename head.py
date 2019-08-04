# Copyright <2019> <Chen Wang <https://chenwang.site>, Carnegie Mellon University>

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
        _,C,H,W = memory.size()
        self.embeddings_size = C*H*W

    def is_read_head(self):
        return NotImplementedError

    def _address_memory(self, key, strength, sharpen):
        # Handle Activations
        key = key.clone()
        strength = F.softplus(strength)
        sharpen = 1 + F.softplus(sharpen)
        return self.memory.address(key, strength, sharpen)


class ReadHead(HeadBase):
    def __init__(self, memory):
        super(ReadHead, self).__init__(memory)
        self.strength = nn.Linear(self.embeddings_size, 1)
        self.sharpen = nn.Linear(self.embeddings_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.strength.weight, gain=1.4)
        nn.init.normal_(self.strength.bias, std=0.01)
        nn.init.xavier_uniform_(self.sharpen.weight, gain=1.4)
        nn.init.normal_(self.sharpen.bias, std=0.01)

    def is_read_head(self):
        return True

    def forward(self, embeddings):
        coding = embeddings.view(embeddings.size(0),-1)
        strength, sharpen = self.strength(coding), self.sharpen(coding)
        w = self._address_memory(embeddings, strength, sharpen)
        return self.memory.read(w)


class WriteHead(HeadBase):
    def __init__(self, memory):
        super(WriteHead, self).__init__(memory)
        self.strength = nn.Linear(self.embeddings_size, 1)
        self.sharpen = nn.Linear(self.embeddings_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.strength.weight, gain=1.4)
        nn.init.normal_(self.strength.bias, std=0.01)
        nn.init.xavier_uniform_(self.sharpen.weight, gain=1.4)
        nn.init.normal_(self.sharpen.bias, std=0.01)

    def is_read_head(self):
        return False

    def forward(self, embeddings):
        coding = embeddings.view(embeddings.size(0),-1)
        strength, sharpen = self.strength(coding), self.sharpen(coding)
        w = self._address_memory(embeddings, strength, sharpen)
        self.memory.write(w, embeddings)


if __name__ == "__main__":
    from memory import Memory
    N, C, H, W = 2000, 2, 2, 2
    B = 2
    memory = Memory(N, C, H, W)
    coding = torch.rand(B, C, H, W)
    batch  = nn.BatchNorm2d(C)
    coding = batch(coding)
    writer = WriteHead(memory)
    reader = ReadHead(memory)

    for i in range(10):
        writer(coding)
        x = reader(coding)
