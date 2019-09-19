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

import cv2
import math
import torch
import random
import numbers
import collections
import torchvision
from torch import nn
from itertools import repeat
from torch import rfft, irfft
import torch.nn.functional as F
import torchvision.transforms as transforms


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class ConvLoss(nn.Module):
    def __init__(self, kernel_size, in_channels=3):
        super(ConvLoss, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=kernel_size, stride=kernel_size//2, bias=False)
        self.pool1 = nn.AvgPool2d(2)
        self.pool2 = nn.AvgPool2d(3)
        self.pool3 = nn.MaxPool2d(3)
        self.conv1.weight.data = torch.ones(self.conv1.weight.size())/self.conv1.weight.numel()

    def forward(self, x, y):
        output = self.pool1(self.conv1((x-y).abs()))
        return self.pool3(self.pool2(output)).squeeze()

class CosineLoss(nn.CosineEmbeddingLoss):
    def __init__(self, dim=1):
        super(CosineLoss, self).__init__()
        self.target = torch.ones(dim).cuda()

    def forward(self, x, y):
        return super(CosineLoss, self).forward(x, y, self.target)


class Split2d(nn.Module):
    def __init__(self, kernel_size=(3, 3)):
        super(Split2d, self).__init__()
        self.h, self.w = _pair(kernel_size)
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        output = self.unfold(x).view(x.size(0), x.size(1), self.h, self.w, -1)
        return output.permute(0,4,1,2,3).contiguous().view(-1, x.size(1), self.h, self.w)


class Merge2d(nn.Module):
    def __init__(self, output_size, kernel_size):
        super(Merge2d, self).__init__()
        self.H, self.W = _pair(output_size)
        self.h, self.w = _pair(kernel_size)
        self.fold = nn.Fold(output_size, kernel_size, stride=kernel_size)

    def forward(self, x):
        output = x.view(-1, (self.H//self.h)*(self.W//self.w), x.size(1)*self.h*self.w)
        return self.fold( output.permute(0,2,1).contiguous())


class RandomMotionBlur(object):
    def __init__(self, p=[0.7, 0.2, 0.1]):
        self.p = p
        kernel_size = 3
        self.w3 = torch.zeros(4, kernel_size, kernel_size)
        self.w3[0,kernel_size//2,:] = 1.0/kernel_size
        self.w3[1,:,kernel_size//2] = 1.0/kernel_size
        self.w3[2] = torch.eye(kernel_size)
        self.w3[3] = torch.eye(kernel_size).rot90()
        kernel_size = 5
        self.w5 = torch.zeros(4, kernel_size, kernel_size)
        self.w5[0,kernel_size//2,:] = 1.0/kernel_size
        self.w5[1,:,kernel_size//2] = 1.0/kernel_size
        self.w5[2] = torch.eye(kernel_size)
        self.w5[3] = torch.eye(kernel_size).rot90()

    def __call__(self, img):
        """
        Args:
            tensor (Image): Image to be cropped.

        Returns:
            tensor: Random motion blured image.
        """
        p = random.random()
        if p <= self.p[0]:
            return img
        if self.p[0] < p <= self.p[0]+ self.p[1]:
            w = self.w3[torch.randint(0,4,(1,))].unsqueeze(0)
            kernel_size = 3
        elif 1-self.p[2] < p:
            w = self.w5[torch.randint(0,4,(1,))].unsqueeze(0)
            kernel_size = 5

        return F.conv2d(img.unsqueeze(1), w, padding=kernel_size//2).squeeze(1)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class EarlyStopScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                    verbose=False, threshold=1e-4, threshold_mode='rel',
                    cooldown=0, min_lr=0, eps=1e-8):
        super().__init__(optimizer, mode, factor, patience, verbose, threshold, threshold_mode, cooldown, min_lr, eps)
        self.no_decrease = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            return self._reduce_lr(epoch)

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
                return False
            else:
                return True


class CorrelationSimilarity(nn.Module):
    '''
    Correlation Similarity for multi-channel 2-D patch via FFT
    args: input_size: tuple(H, W) --> size of last two dimensions
    Input Shape:
    x: tensor(B, C, H, W)
    y: tensor(N, C, H, W)
    Output Shape:
    o: tensor(B, N)
    '''
    def __init__(self, input_size):
        super(CorrelationSimilarity, self).__init__()
        input_size = _pair(input_size)
        assert(input_size[-1]!=1) # FFT2 is wrong if last dimension is 1
        self.N = math.sqrt(input_size[0]*input_size[1])
        self.fft_args = {'signal_ndim':2, 'normalized':True, 'onesided': True}
        self.ifft_args = {**self.fft_args, **{'signal_sizes':input_size}}
        self.max = nn.MaxPool2d(kernel_size=input_size)
        g = torch.zeros(input_size).view(1,1,input_size[0], input_size[1]).cuda()
        g[0,0,0,0] = 1
        self.G = rfft(g, **self.fft_args)

    def forward(self, x, y):
        X = x.rfft(**self.fft_args).unsqueeze(1)
        Y = y.rfft(**self.fft_args)
        g = cdot(conj(X), Y).sum(dim=2).irfft(**self.ifft_args)*self.N
        xx = x.view(x.size(0),-1).norm(dim=-1).view(x.size(0), 1)
        yy = y.view(y.size(0),-1).norm(dim=-1).view(1, y.size(0))
        return self.max(g).view(x.size(0), y.size(0))/xx/yy


def cdot(X, Y):
    '''
    complex dot multiplication
    '''
    assert(X.size(-1)==Y.size(-1)==2)
    SX, SY = X.size(), Y.size()
    X, Y = X.view(-1,2), Y.view(-1,2)
    A, B = X[:,0].view(SX[:-1]), X[:,1].view(SX[:-1])
    C, D = Y[:,0].view(SY[:-1]), Y[:,1].view(SY[:-1])
    return torch.stack((A*C - B*D, B*C + A*D), dim=-1).contiguous()


def conj(X):
    '''
    complex conjugate
    '''
    assert(X.size(-1)==2)
    SX = X.size()
    X = X.view(-1,2)
    A, B = X[:,0].view(SX[:-1]), X[:,1].view(SX[:-1])
    return torch.stack((A, -B), dim=-1).contiguous()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_batch(batch, name='video'):
    min_v = torch.min(batch)
    range_v = torch.max(batch) - min_v
    if range_v > 0:
        batch = (batch - min_v) / range_v
    else:
        batch = torch.zeros(batch.size())
    grid = torchvision.utils.make_grid(batch, padding=0).cpu()
    img = grid.numpy()[::-1].transpose((1, 2, 0))
    cv2.imshow(name, img)
    cv2.waitKey(1)
    return img


def show_batch_origin(batch, name='video'):
    grid = torchvision.utils.make_grid(batch).cpu()
    img = grid.numpy()[::-1].transpose((1, 2, 0))
    cv2.imshow(name, img)
    cv2.waitKey(1)


if __name__ == "__main__":
    motionblur = RandomMotionBlur()
    x = torch.randn(15, 512, 12, 12)
    split = Split2d((3,3))
    merge = Merge2d((12, 12), (3,3))
    s = split(x)
    o = merge(s)
    print(x.shape, s.shape, o.shape)
    criterion = ConvLoss(kernel_size=32, in_channels=3)
    x = torch.randn(10, 3, 320, 320)
    y = torch.randn(10, 3, 320, 320)
    loss = criterion(x, y)
    print(loss.shape)

    C, M, N = 1, 3, 3
    similarity = CorrelationSimilarity((M,N))
    x = torch.randn(1, C, M, N).cuda()
    y = torch.randn(2, C, M, N).cuda()
    y[1,:,:,:] = x
    k = similarity(x,y)
    s = F.softmax(k,dim=1)
    print(k)
    print(s)