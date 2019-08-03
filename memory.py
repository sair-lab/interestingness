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
    def __init__(self, N=2000, M=512):
        """Initialize the Memory matrix.
        N: Number of rows in the memory.
        M: Number of columns/features in the memory.
        """
        super(Memory, self).__init__()
        self.N,self.M = N, M
        self.register_buffer('memory', torch.Tensor(N, M))
        self.reset_parameters()

    def reset_parameters(self):
        stdev = 1 / (np.sqrt(self.N + self.M))
        nn.init.uniform_(self.memory, -stdev, stdev)

    def size(self):
        return self.N, self.M

    def read(self, w):
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.N, self.M)
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

    def address(self, key, strength, sharpen):
        """NTM Addressing (according to section 3.3).
        Returns a softmax weighting over the rows of the memory matrix.

        key: The key vector.
        strength: The key strength (focus).
        sharpen: Sharpen weighting scalar.
        """
        # Content focus
        w = self._similarity(key, strength)
        w = self._sharpen(w, sharpen)
        return w

    def _similarity(self, key, strength):
        key = key.unsqueeze(1)
        w = F.softmax(strength * F.cosine_similarity(self.memory + 1e-16, key + 1e-16, dim=-1), dim=1)
        return w

    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w
