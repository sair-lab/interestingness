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
import tqdm
import torch
import os.path
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision.models.vgg import VGG
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import ImageData, Dronefilm
from interestingness import AE, VAE, Interestingness
from torchutil import EarlyStopScheduler, count_parameters, show_batch, RandomMotionBlur, CosineLoss


def train(loader, net):
    train_loss, batches = 0, len(loader)
    enumerater = tqdm.tqdm(enumerate(loader))
    for batch_idx, inputs in enumerater:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        optimizer.zero_grad()
        inputs = Variable(inputs)
        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        enumerater.set_description("train loss: %.4f on %d/%d"%(train_loss/(batch_idx+1), batch_idx, batches))

    return train_loss/(batch_idx+1)


def performance(loader, net):
    test_loss = 0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            inputs = Variable(inputs).view(-1,inputs.size(-3),inputs.size(-2),inputs.size(-1))
            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            test_loss += loss.item()
            show_batch(torch.cat([inputs,outputs], dim=0), name='train')

    return test_loss/(batch_idx+1)


def test(loader, net):
    test_loss = 0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            inputs = Variable(inputs).view(-1,inputs.size(-3),inputs.size(-2),inputs.size(-1))
            outputs = net.listen(inputs)
            loss = criterion(outputs, inputs)
            test_loss += loss.item()
            show_batch(torch.cat([inputs,outputs], dim=0), name='test')

    return test_loss/(batch_idx+1)


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Train Interestingness Networks')
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset root folder")
    parser.add_argument("--model-save", type=str, default='saves/ae.pt', help="learning rate")
    parser.add_argument("--data", type=str, default='car', help="training data name")
    parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
    parser.add_argument("--factor", type=float, default=0.1, help="ReduceLROnPlateau factor")
    parser.add_argument("--min-lr", type=float, default=1e-1, help="minimum lr for ReduceLROnPlateau")
    parser.add_argument("--patience", type=int, default=10, help="patience of epochs for ReduceLROnPlateau")
    parser.add_argument("--epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="number of minibatch size")
    parser.add_argument("--momentum", type=float, default=0, help="momentum of the optimizer")
    parser.add_argument("--alpha", type=float, default=0.1, help="weight of TVLoss")
    parser.add_argument("--w-decay", type=float, default=1e-2, help="weight decay of the optimizer")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.set_defaults(self_loop=False)
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
            # transforms.RandomRotation(20),
            transforms.RandomCrop(320),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    train_data = Dronefilm(root=args.data_root, train=True,  data=args.data, transform=transform)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

    net,_ = torch.load(args.model_save)
    net = Interestingness(net, 4000, 512, 10, 10, 10, 10)
    net.set_train(True)

    if torch.cuda.is_available():
        net = net.cuda()

    # criterion = nn.MSELoss()
    criterion = CosineLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr, patience=args.patience)

    print('number of parameters:', count_parameters(net))
    best_loss = float('Inf')
    for epoch in range(args.epochs):
        train_loss = performance(train_loader, net)
        val_loss = test(train_loader, net)
        print('epoch:{} train:{} val:{}'.format(epoch, train_loss, val_loss))

        if val_loss < best_loss:
            print("New best Model, saving...")
            torch.save(net, args.model_save+'.corr.read.interest.'+ args.data)
            best_loss = val_loss
            no_decrease = 0
                
        if scheduler.step(val_loss, epoch):
            print("Early Stopping!")
            break

    print('test_loss, %.4f'%(best_loss))
