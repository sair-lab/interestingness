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
from matplotlib import pyplot as plt
from torchvision.models.vgg import VGG
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import ImageData, Dronefilm
from torchutil import count_parameters, show_batch
from interestingness import AE, VAE, AutoEncoder, Interestingness


def performance(loader, net):
    test_loss = 0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            if batch_idx % 10 !=0:
                continue
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            inputs = Variable(inputs).view(-1,inputs.size(-3),inputs.size(-2),inputs.size(-1))
            outputs= net(inputs)
            loss = max([criterion(outputs[i], inputs[i]) for i in range(inputs.size(0))])
            test_loss += loss.item()
            show_batch_loss(inputs, batch_idx, loss.item())
            show_batch(inputs-outputs, 'head-map')
            print('loss:', loss.item())

    return test_loss/(batch_idx+1)


def show_batch_loss(batch, batch_idx, loss):
    min_v = torch.min(batch)
    range_v = torch.max(batch) - min_v
    if range_v > 0:
        batch = (batch - min_v) / range_v
    else:
        batch = torch.zeros(batch.size())
    grid = torchvision.utils.make_grid(batch)
    grid = grid.cpu()
    plt.subplot(121)
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))
    plt.title(str(batch_idx))
    plt.subplot(122)
    plt.bar(0, loss, width=0.1)
    plt.plot([-0.05,0.05], [1.5, 1.5], 'r')
    plt.plot([-0.05,0.05], [1, 1], 'r')
    plt.axis('equal')
    plt.xlim(-0.05, 0.05)
    plt.ylim(0, 2)
    plt.draw()
    plt.pause(.1)
    plt.clf()


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset root folder")
    parser.add_argument("--model-save", type=str, default='saves/ae.pt.interest', help="learning rate")
    parser.add_argument("--data", type=str, default='car', help="training data name")
    parser.add_argument("--batch-size", type=int, default=1, help="number of minibatch size")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.set_defaults(self_loop=False)
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # transform = transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomResizedCrop((384, 640)),
    #         transforms.FiveCrop(384),
    #         transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    #         transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    #         ])

    test_data = Dronefilm(root=args.data_root, train=False,  data=args.data, test_id=0, transform=transform)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    net = torch.load(args.model_save+'.'+args.data)
    net.set_train(False)

    if torch.cuda.is_available():
        net = net.cuda()

    criterion = nn.MSELoss()

    print('number of parameters:', count_parameters(net))

    val_loss = performance(test_loader, net)
