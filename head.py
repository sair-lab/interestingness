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


def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


class HeadBase(nn.Module):

    def __init__(self, memory, controller_size):
        """Initilize the read/write head.

        :param memory: The :class:`Memory` to be addressed by the head.
        :param controller_size: The size of the internal representation.
        """
        super(HeadBase, self).__init__()

        self.memory = memory
        self.N, self.M = memory.size()
        self.controller_size = controller_size

    def create_new_state(self, batch_size):
        raise NotImplementedError

    def register_parameters(self):
        raise NotImplementedError

    def is_read_head(self):
        return NotImplementedError

    def _address_memory(self, key, strength, sharpen):
        # Handle Activations
        key = key.clone()
        strength = torch.softplus(strength)
        sharpen = 1 + torch.softplus(sharpen)
        return self.memory.address(key, strength, sharpen)


class ReadHead(HeadBase):
    def __init__(self, memory, controller_size):
        super(ReadHead, self).__init__(memory, controller_size)

        # Corresponding to key, strength, sharpen sizes
        self.read_lengths = [self.M, 1, 1]
        self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        # The state holds the previous time step address weightings
        return torch.zeros(batch_size, self.N)

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_read.weight, gain=1.4)
        nn.init.normal_(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True

    def forward(self, embeddings):
        """ReadHead forward function.

        param embeddings: input representation of the controller.
        """
        o = self.fc_read(embeddings)
        key, strength, sharpen = _split_cols(o, self.read_lengths)

        # Read from memory
        w = self._address_memory(key, strength, sharpen)
        r = self.memory.read(w)

        return r


class WriteHead(HeadBase):
    def __init__(self, memory, controller_size):
        super(WriteHead, self).__init__(memory, controller_size)

        # Corresponding to key, strength, sharpen, earse, add sizes
        self.write_lengths = [self.M, 1, 1, self.M, self.M]
        self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform_(self.fc_write.weight, gain=1.4)
        nn.init.normal_(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    def forward(self, embeddings, w_prev=None):
        """WriteHead forward function.

        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        o = self.fc_write(embeddings)
        key, strength, sharpen, earse, add = _split_cols(o, self.write_lengths)

        # e should be in [0, 1]
        earse = torch.sigmoid(earse)

        # Write to memory
        w = self._address_memory(key, strength, sharpen)
        self.memory.write(w, earse, add)

        return w
