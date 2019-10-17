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

import os
import cv2
import copy
import time
import tqdm
import torch
import os.path
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.utils.data as Data
from torch.autograd import Variable
from torch.nn import functional as F
from matplotlib import pyplot as plt
from torchvision.models.vgg import VGG
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import ImageData, Dronefilm, SubT
from interestingness import AE, VAE, AutoEncoder, Interestingness
from torchutil import count_parameters, show_batch, ConvLoss, CosineLoss, CorrelationLoss, Split2d, Merge2d, PearsonLoss, FiveSplit2d

# 2019-10-10, yf
from torch.utils.tensorboard import SummaryWriter
logger = SummaryWriter('runs/')

class Interest():
    '''
    Maintain top K interests
    '''
    def __init__(self, K):
        self.K = K
        self.interests = []

    def add_interest(self, tensor, loss, visualize_window=None):
        self.interests.append((loss, tensor))
        self.interests.sort(key=self._sort_loss, reverse=True)
        self._maintain()
        interests = np.concatenate([self.interests[i][1] for i in range(len(self.interests))], axis=0)
        if visualize_window is not None:
            cv2.imshow(visualize_window, interests)
        return interests

    def _sort_loss(self, val):
        return val[0]

    def _maintain(self):
        if len(self.interests) > self.K:
            self.interests = self.interests[:self.K]


def performance(loader, net):
    test_loss = 0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            if batch_idx % 10 !=0:
                continue
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            outputs= net(inputs)
            loss = criterion(fivecrop(inputs), fivecrop(outputs)).max()
            drawbox(inputs, outputs)
            test_loss += loss.item()
            image = show_batch(torch.cat([outputs, (outputs-inputs).abs()], dim=0), 'reconstruction')
            frame = show_batch_box(inputs, batch_idx, loss.item())
            interest.add_interest(frame, loss, visualize_window='Top Interests')
            cv2.imwrite('images/interestingness-convmse-%04d.png'%(batch_idx), 255*np.concatenate([frame, image], axis=1))
            print('batch_idx:', batch_idx, 'loss:%.6f'%(loss.item()))
            logger.add_scalar('loss', loss.item(), batch_idx)
    cv2.waitKey(0)
    return test_loss/(batch_idx+1)


def boxbar(height, bar, ranges=[0, 1], threshold=[0.8, 0.9]):
    width = 15
    box = np.zeros((height,width,3), np.uint8)
    x1, y1 = 0, int((1.0-bar/(ranges[1]-ranges[0]))*height)
    x2, y2 = int(width), int(height)
    cv2.rectangle(box,(x1,y1),(x2,y2),(0,1,0),-1)
    for i in threshold:
        x1, y1 = 0, int((1.0-i/ranges[1])*height)
        x2, y2 = width, int((1.0-i/ranges[1])*height)
        cv2.line(box,(x1, y1), (x2, y2), (1,0,0), 3)
    return box


def show_batch_box(batch, batch_idx, loss, box_id=None):
    min_v = torch.min(batch)
    range_v = torch.max(batch) - min_v
    if range_v > 0:
        batch = (batch - min_v) / range_v
    else:
        batch = torch.zeros(batch.size())
    grid = torchvision.utils.make_grid(batch).cpu()
    img = grid.numpy()[::-1].transpose((1, 2, 0))
    box = boxbar(grid.size(-2), loss)
    frame = np.hstack([img, box])
    cv2.imshow('interestingness', frame)
    cv2.waitKey(1)
    return frame


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Test Interestingness Networks')
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset root folder")
    parser.add_argument("--model-save", type=str, default='saves/ae.pt.corr.read.interest', help="learning rate")
    parser.add_argument("--data", type=str, default='car', help="training data name")
    parser.add_argument("--batch-size", type=int, default=1, help="number of minibatch size")
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument("--crop-size", type=int, default=320, help='loss compute by grid')
    parser.add_argument("--num-interest", type=int, default=3, help='loss compute by grid')
    parser.set_defaults(self_loop=False)
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # test_data = Dronefilm(root=args.data_root, train=False,  data=args.data, test_id=0, transform=transform)
    test_data = SubT(root=args.data_root, train=False, transform=transform)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    net = torch.load(args.model_save+'.'+args.data)
    net.set_train(False)

    interest = Interest(args.num_interest)
    if torch.cuda.is_available():
        net = net.cuda()

    drawbox = ConvLoss(input_size=args.crop_size, kernel_size=args.crop_size//2, stride=args.crop_size//4)
    criterion = CorrelationLoss(args.crop_size//2, reduce=False, accept_translation=False)
    fivecrop = FiveSplit2d(args.crop_size//2)

    print('number of parameters:', count_parameters(net))

    val_loss = performance(test_loader, net)
