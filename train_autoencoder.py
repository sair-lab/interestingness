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
from autoencoder import VGGNet, AEs, AE8x, AE8s, AE32s, AE16s


def train(loader, net):
    scheduler.step()
    train_loss, batches = 0, len(loader)
    enumerater = tqdm.tqdm(enumerate(loader))
    for batch_idx, (inputs, _) in enumerater:
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
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            outputs = net(inputs)
            loss = criterion(outputs, outputs)
            test_loss += loss.item()

    return test_loss/(batch_idx+1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--data-root", type=str, default='/data/datasets/coco', help="dataset root folder")
    parser.add_argument("--annFile", type=str, default='/data/datasets/coco', help="learning rate")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.1, help="learning rate multiplier")
    parser.add_argument("--milestones", type=int, default=100, help="milestones for applying multiplier")
    parser.add_argument("--epochs", type=int, default=150, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=50, help="number of minibatch size")
    parser.add_argument("--momentum", type=float, default=0, help="momentum of the optimizer")
    parser.add_argument("--w-decay", type=float, default=1e-5, help="weight decay of the optimizer")
    parser.add_argument('--seed', type=int, default=3, help='Random seed.')
    parser.set_defaults(self_loop=False)
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_root = os.path.join(args.data_root, 'images/train2017')
    val_root = os.path.join(args.data_root, 'images/val2017')
    test_root = os.path.join(args.data_root, 'images/test2017')

    train_annFile = os.path.join(args.annFile, 'annotations/annotations_trainval2017/captions_train2017.json')
    val_annFile = os.path.join(args.annFile, 'annotations/annotations_trainval2017/captions_val2017.json')
    test_annFile = os.path.join(args.annFile, 'annotations/image_info_test2017/image_info_test2017.json')   

    train_data = CocoDetection(root=train_root, annFile=train_annFile, transform=train_transform)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

    val_data = CocoDetection(root=val_root, annFile=val_annFile, transform=val_transform)
    val_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)

    net = AEs(pretrained_net=VGGNet(requires_grad=True), n_class=3).cuda()
    net = nn.DataParallel(net, device_ids=list(range(torch.cuda.device_count())))

    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.milestones], gamma=args.gamma)


    print('number of parameters:', count_parameters(net))
    best_loss = 1e10
    for epoch in range(args.epochs):
        train_loss = train(train_loader, net)
        val_loss = performance(val_loader, net) # validate
        print("epoch: %d, train_loss: %.4f, val_loss: %.4f, val_acc: %.2f" % (epoch, train_loss, val_loss))

        if val_loss < best_loss:
            print("New best Model, saving...")
            torch.save(net, "saves/model.pt")
            best_loss = val_loss

    net = torch.load("saves/model.pt")
    test_data = CocoDetection(root=test_root, annFile=test_annFile, transform=val_transform)
    test_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)

    test_loss, test_acc = performance(net, test_loader)
    print('val_acc: %.2f, test_loss, %.4f test_acc: %.2f'%(best_loss, test_loss, test_acc))
