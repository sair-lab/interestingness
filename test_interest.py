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

from dataset import ImageData, Dronefilm, DroneFilming, SubT, SubTF, PersonalVideo
from interestingness import AE, VAE, AutoEncoder, Interestingness
from torchutil import count_parameters, show_batch, show_batch_origin, ConvLoss, CosineLoss, CorrelationLoss, Split2d, Merge2d, PearsonLoss, FiveSplit2d

class Interest():
    '''
    Maintain top K interests
    '''
    def __init__(self, K, filename):
        self.K = K
        self.interests = []
        self.filename = filename
        f = open(self.filename, 'w')
        f.close()

    def add_interest(self, tensor, loss, batch_idx, visualize_window=None):
        f = open(self.filename, 'a+')
        f.write("%d %f\n" % (batch_idx, loss))
        f.close()
        self.interests.append((loss, tensor, batch_idx))
        self.interests.sort(key=self._sort_loss, reverse=True)
        self._maintain()
        interests = np.concatenate([self.interests[i][1] for i in range(len(self.interests))], axis=1)
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
            if batch_idx % args.skip_frames !=0:
                continue
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            outputs, loss = net(inputs)
            if args.debug is not True:
                drawbox(inputs, outputs)
            test_loss += loss.item()
            frame = show_batch_box(inputs, batch_idx, loss.item())
            top_interests = interest.add_interest(frame, loss, batch_idx, visualize_window='Top Interests')
            if args.debug is True:
                image = show_batch(torch.cat([outputs], dim=0), 'reconstruction')
                recon = show_batch(torch.cat([(inputs-outputs).abs()], dim=0), 'difference')
                cv2.imwrite('images/%s-%d/%s-interestingness-%06d.png'%(args.dataset,args.test_data,args.save_flag,batch_idx), frame*255)
                cv2.imwrite('images/%s-%d/%s-reconstruction-%06d.png'%(args.dataset,args.test_data,args.save_flag,batch_idx), image*255)
                cv2.imwrite('images/%s-%d/%s-difference-%06d.png'%(args.dataset,args.test_data,args.save_flag,batch_idx), recon*255)
            print('batch_idx:', batch_idx, 'loss:%.6f'%(loss.item()))

    cv2.imwrite('results/%s.png'%(test_name), 255*top_interests)
    return test_loss/(batch_idx+1)


def boxbar(height, bar, ranges=[0, 0.1], threshold=[0.05, 0.06]):
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
    box = boxbar(grid.size(-2), loss, threshold=[])
    frame = np.hstack([img, box])
    cv2.imshow('interestingness', frame)
    cv2.waitKey(1)
    return frame


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Test Interestingness Networks')
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset root folder")
    parser.add_argument("--model-save", type=str, default='saves/ae.pt.DroneFilming.interest.mse', help="learning rate")
    parser.add_argument("--test-data", type=int, default=0, help='test data ID.')
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument("--crop-size", type=int, default=320, help='loss compute by grid')
    parser.add_argument("--num-interest", type=int, default=10, help='loss compute by grid')
    parser.add_argument("--skip-frames", type=int, default=1, help='skip frame')
    parser.add_argument('--dataset', type=str, default='PersonalVideo', help='dataset type (subT ot drone')
    parser.add_argument('--save-flag', type=str, default='interests', help='save name flat')
    parser.add_argument("--rr", type=float, default=5, help="reading rate")
    parser.add_argument("--wr", type=float, default=5, help="reading rate")
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)

    if not os.path.exists('results'):
        os.makedirs('results')

    if args.debug is True and not os.path.exists('images/%s-%d'%(args.dataset,args.test_data)):
        os.makedirs('images/%s-%d'%(args.dataset,args.test_data))

    transform = transforms.Compose([
            # transforms.CenterCrop(args.crop_size),
            transforms.Resize((args.crop_size,args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    test_name = '%s-%d-%s-%s'%(args.dataset, args.test_data, time.strftime('%Y-%m-%d-%H:%M:%S'), args.save_flag)

    if args.dataset == 'DroneFilming':
        test_data = DroneFilming(root=args.data_root, train=False, test_data=args.test_data, transform=transform)
    elif args.dataset == 'SubTF':
        test_data = SubTF(root=args.data_root, train=False, test_data=args.test_data, transform=transform)
    elif args.dataset == 'PersonalVideo':
        test_data = PersonalVideo(root=args.data_root, train=False, test_data=args.test_data, transform=transform)

    test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    net = torch.load(args.model_save)
    net.set_train(False)
    net.memory.set_learning_rate(rr=args.rr, wr=args.wr)

    interest = Interest(args.num_interest, 'results/%s.txt'%(test_name))
    if torch.cuda.is_available():
        net = net.cuda()

    drawbox = ConvLoss(input_size=args.crop_size, kernel_size=args.crop_size//2, stride=args.crop_size//4)
    criterion = CorrelationLoss(args.crop_size//2, reduce=False, accept_translation=False)
    fivecrop = FiveSplit2d(args.crop_size//2)

    print('number of parameters:', count_parameters(net))
    val_loss = performance(test_loader, net)
    print('Done.')
