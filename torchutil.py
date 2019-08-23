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
import torch
import random
import numbers
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms


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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_batch(batch, name='video'):
    min_v = torch.min(batch)
    range_v = torch.max(batch) - min_v
    if range_v > 0:
        batch = (batch - min_v) / range_v
    else:
        batch = torch.zeros(batch.size())
    grid = torchvision.utils.make_grid(batch).cpu()
    img = grid.numpy()[::-1].transpose((1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow(name, img)
    cv2.waitKey(1)

if __name__ == "__main__":
    motionblur = RandomMotionBlur()