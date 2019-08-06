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
from autoencoder import VGGNet, AEs, AE8x, AE8s, AE32s, AE16s


def performance(loader, net):
    test_loss = 0
    with torch.no_grad():
        enumerater = tqdm.tqdm(enumerate(loader))

        for batch_idx, (inputs, _) in enumerater:
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            test_loss += loss.item()
            show_batch(torch.cat([inputs,outputs], dim=0).cpu())

    return test_loss/(batch_idx+1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_batch(batch):
    grid = torchvision.utils.make_grid(batch)
    plt.imshow(grid.numpy()[::-1].transpose((1, 2, 0)))
    plt.title('Batch from dataloader')
    plt.show()

if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Feature Graph Networks')
    parser.add_argument("--data-root", type=str, default='/data/datasets/coco', help="dataset root folder")
    parser.add_argument("--annFile", type=str, default='/data/datasets/coco', help="learning rate")
    parser.add_argument("--model-save", type=str, default='saves/model.pt', help="learning rate")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.1, help="learning rate multiplier")
    parser.add_argument("--milestones", type=int, default=100, help="milestones for applying multiplier")
    parser.add_argument("--epochs", type=int, default=150, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="number of minibatch size")
    parser.add_argument("--momentum", type=float, default=0, help="momentum of the optimizer")
    parser.add_argument("--w-decay", type=float, default=1e-5, help="weight decay of the optimizer")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.set_defaults(self_loop=False)
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)


    val_transform = transforms.Compose([
            transforms.RandomResizedCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    test_root = os.path.join(args.data_root, 'images/test2017')
    test_annFile = os.path.join(args.annFile, 'annotations/image_info_test2017/image_info_test2017.json')   

    net = torch.load("saves/ae-384.pt")
    net.eval()
    criterion = nn.MSELoss()

    print('number of parameters:', count_parameters(net))
    test_data = CocoDetection(root=test_root, annFile=test_annFile, transform=val_transform)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    print('len of loader', len(test_data))
    test_loss = performance(test_loader, net)
    print('test_loss, %.4f'%(test_loss))
