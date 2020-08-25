#!/usr/bin/env python3

import os
import cv2
import copy
import time
import math
import torch
import os.path
import argparse
import torchvision
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

from dataset import DroneFilming, SubTF
from interestingness import Interestingness
from torchutil import ConvLoss, CosineLoss, CorrelationLoss
from torchutil import count_parameters, show_batch, Timer, MovAvg


class Interest():
    '''
    Maintain top K interests
    '''
    def __init__(self, K):
        self.K = K
        self.interests = []

    def add_interest(self, tensor, loss, batch_idx, visualize_window=None):
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


def performance(loader, net, args):
    time_use, timer = 0, Timer()
    movavg = MovAvg(args.window_size)
    test_name = '%s-%d-%s-%s'%(args.dataset, args.test_data, time.strftime('%Y-%m-%d-%H:%M:%S'), args.save_flag)
    file_name = 'results/%s.txt'%(test_name)
    interest = Interest(args.num_interest)
    drawbox = ConvLoss(input_size=args.crop_size, kernel_size=args.crop_size//2, stride=args.crop_size//4)

    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            if batch_idx % args.skip_frames !=0:
                continue
            inputs = inputs.to(args.device)
            timer.tic()
            loss = net(inputs)
            loss = movavg.append(loss)
            time_use += timer.end()
            with open(file_name, 'a+') as f:
                f.write("%d %f\n" % (batch_idx, loss))
            if args.noshow is False:
                outputs = net.output()
            if args.drawbox is True and not args.noshow:
                drawbox(inputs, outputs)
            if args.noshow is False:
                frame = show_batch_box(inputs, batch_idx, loss.item())
                top_interests = interest.add_interest(frame, loss, batch_idx, visualize_window='Top Interests')
            if args.debug is True and not args.noshow:
                debug = show_batch(torch.cat([outputs, (inputs-outputs).abs()], dim=0), 'debug')
                debug = np.concatenate([frame, debug], axis=1)
                cv2.imwrite('images/%s-%d/%s-debug-%06d.png'%(args.dataset,args.test_data,args.save_flag,batch_idx), debug*255)
            print('batch_idx:', batch_idx, 'loss:%.6f'%(loss.item()))

    if args.noshow is False:
        cv2.imwrite('results/%s.png'%(test_name), 255*top_interests)
    print("Total time using: %.2f seconds, %.2f ms/frame"%(time_use, 1000*time_use/(batch_idx+1)))


def level_height(bar, ranges=[0.02, 0.08]):
    h = min(max(0,(bar-ranges[0])/(ranges[1]-ranges[0])),1)
    return (np.tanh(np.tan(math.pi/2*(2*h-1))-0.8)+1)/2


def boxbar(height, bar, ranges=[0.02, 0.08], threshold=[0.05, 0.06]):
    width = 15
    box = np.zeros((height,width,3), np.uint8)
    h = level_height(bar, ranges)
    x1, y1 = 0, int((1-h)*height)
    x2, y2 = int(width), int(height)
    cv2.rectangle(box,(x1,y1),(x2,y2),(0,1,0),-1)
    for i in threshold:
        x1, y1 = 0, int((1.0-i/ranges[1])*height)
        x2, y2 = width, int((1.0-i/ranges[1])*height)
        cv2.line(box,(x1, y1), (x2, y2), (1,0,0), 3)
    return box


def show_batch_box(batch, batch_idx, loss, box_id=None, show_now=True):
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
    if show_now:
        cv2.imshow('interestingness', frame)
        cv2.waitKey(1)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Interestingness Networks')
    parser.add_argument('--device', type=str, default='cuda', help='cpu, cuda:0, cuda:1, etc.')
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset root folder")
    parser.add_argument("--model-save", type=str, default='saves/vgg16.pt.SubTF.n100usage.mse', help="read model")
    parser.add_argument("--test-data", type=int, default=2, help='test data ID.')
    parser.add_argument("--seed", type=int, default=0, help='Random seed.')
    parser.add_argument("--crop-size", type=int, default=320, help='crop size')
    parser.add_argument("--num-interest", type=int, default=10, help='loss compute by grid')
    parser.add_argument("--skip-frames", type=int, default=1, help='number of skip frame')
    parser.add_argument("--window-size", type=int, default=1, help='smooth window size >=1')
    parser.add_argument('--dataset', type=str, default='SubTF', help='dataset type (SubTF, DroneFilming')
    parser.add_argument('--save-flag', type=str, default='n100usage', help='save name flag')
    parser.add_argument("--rr", type=float, default=5, help="reading rate")
    parser.add_argument("--wr", type=float, default=5, help="writing rate")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument('--noshow', dest='noshow', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--drawbox', dest='drawbox', action='store_true')
    parser.set_defaults(noshow=False, debug=False, drawbox=False)
    args = parser.parse_args(); print(args)
    datasets = {'dronefilming': DroneFilming, 'subtf': SubTF}
    torch.manual_seed(args.seed)

    os.makedirs('results', exist_ok=True)
    if args.debug is True:
        os.makedirs('images/%s-%d'%(args.dataset,args.test_data), exist_ok=True)

    transform = transforms.Compose([
            # transforms.CenterCrop(args.crop_size),
            transforms.Resize((args.crop_size,args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    Dataset = datasets[args.dataset.lower()]
    test_data = Dataset(root=args.data_root, train=False, test_data=args.test_data, transform=transform)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    net = torch.load(args.model_save).to(args.device)
    net.set_train(False)
    net.memory.set_learning_rate(rr=args.rr, wr=args.wr)

    performance(test_loader, net, args)
    print('number of parameters:', count_parameters(net))
    print('Done.')
