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


class Memory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self, N=2000, C=512, H=7, W=7):
        """Initialize the Memory matrix.
        N: Number of Cubes in the memory.
        C: Channel of cubes in the memory
        H: Height of cubes in the memory
        W: Number of cubes in the memory
        """
        super(Memory, self).__init__()
        self.N, self.C, self.H, self.W = N, C, H, W
        self.register_buffer('memory', torch.Tensor(N, C, H, W))
        self.reset_parameters()

    def reset_parameters(self):
        stdev = 1 / (np.sqrt(self.N*self.C*self.H*self. W))
        nn.init.uniform_(self.memory, -stdev, stdev)

    def size(self):
        return (self.N, self.C, self.H, self.W)

    def read(self, w):
        w = w.view(w.size(0), self.N, 1, 1, 1)
        return torch.sum(w*self.memory,dim=1)

    def write(self, w, add):
        experience = torch.sum(w.view(w.size(0),self.N,1,1,1) * add.unsqueeze(1), dim=0)
        self.memory = self.memory + experience

    def address(self, key, strength, sharpen):
        """
        Returns a softmax weighting over the memory cubes.

        key: The key vector.
        strength: The key strength (focus).
        sharpen: Sharpen weighting scalar.
        """
        # Content focus
        w = self._similarity(key, strength)
        w = self._sharpen(w, sharpen)
        return w

    def _similarity(self, key, strength):
        key = key.view(key.size(0), 1, -1)
        memory = self.memory.view(self.N, -1)
        w = F.softmax(strength * F.cosine_similarity(memory, key, dim=-1), dim=1)
        return w

    def _sharpen(self, w, sharpen):
        w = w ** sharpen
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w


if __name__ == "__main__":
    N, C, H, W = 2000, 512, 7, 7
    B = 2
    mem = Memory(N, C, H, W)
    key = torch.FloatTensor(B, C, H, W)
    t = torch.FloatTensor([0.3]*B).view(B,1)
    s = torch.FloatTensor([0.7]*B).view(B,1)

    w = mem.address(key, t, s)
    x = mem.read(w)
    # e = torch.FloatTensor(B, C, H, W)
    a = torch.FloatTensor(B, C, H, W)
    mem.write(w, a)
    print(x.size())