#!/usr/bin/env python3

import os
import copy
import tqdm
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.transforms as transforms

from autoencoder import AutoEncoder
from interestingness import Interestingness
from dataset import ImageData, Dronefilm, DroneFilming, SubT, SubTF
from torchutil import EarlyStopScheduler, count_parameters, show_batch, RandomMotionBlur, CosineLoss, PearsonLoss


def performance(loader, net, criterion, device='cuda', window='test'):
    test_loss = 0
    with torch.no_grad():
        for batch_idx, inputs in enumerate(loader):
            inputs = inputs.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, inputs)
            test_loss += loss.item()
            show_batch(torch.cat([inputs,outputs], dim=0), name=window)
    return test_loss/(batch_idx+1)


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Train Interestingness Networks')
    parser.add_argument('--device', type=str, default='cuda', help='cpu, cuda:0, cuda:1, etc.')
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset root folder")
    parser.add_argument("--model-save", type=str, default='saves/vgg16.pt', help="learning rate")
    parser.add_argument('--save-flag', type=str, default='n100usage', help='save name flag')
    parser.add_argument("--memory-size", type=int, default=100, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
    parser.add_argument("--factor", type=float, default=0.1, help="ReduceLROnPlateau factor")
    parser.add_argument("--min-lr", type=float, default=1e-1, help="minimum lr for ReduceLROnPlateau")
    parser.add_argument("--patience", type=int, default=2, help="patience of epochs for ReduceLROnPlateau")
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="number of minibatch size")
    parser.add_argument("--momentum", type=float, default=0, help="momentum of the optimizer")
    parser.add_argument("--alpha", type=float, default=0.1, help="weight of TVLoss")
    parser.add_argument("--w-decay", type=float, default=1e-2, help="weight decay of the optimizer")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--loss', type=str, default='mse', help='loss criterion')
    parser.add_argument("--crop-size", type=int, default=320, help='loss compute by grid')
    parser.add_argument("--rr", type=float, default=5, help="reading rate")
    parser.add_argument("--wr", type=float, default=5, help="writing rate")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument('--dataset', type=str, default='SubTF', help='dataset type (subT ot drone')
    args = parser.parse_args(); print(args)
    losses = {'l1': nn.L1Loss, 'mse': nn.MSELoss, 'cos':CosineLoss, 'pearson':PearsonLoss}
    datasets = {'dronefilming': DroneFilming, 'subtf': SubTF}
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
            transforms.RandomCrop(args.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    Dataset = datasets[args.dataset.lower()]
    train_data = Dataset(root=args.data_root, train=True, transform=transform)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    net,_ = torch.load(args.model_save)
    net = Interestingness(net, args.memory_size, 512, 10, 10, 10, 10).to(args.device)
    net.memory.set_learning_rate(args.rr, args.wr)
    net.set_train(True)

    criterion = losses[args.loss.lower()]()
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr, patience=args.patience)

    print('number of parameters:', count_parameters(net))
    best_loss = float('Inf')
    for epoch in range(args.epochs):
        train_loss = performance(train_loader, net, criterion, args.device, 'train')
        val_loss = performance(train_loader, net.listen, criterion, args.device, 'test')
        print('epoch:{} train:{} val:{}'.format(epoch, train_loss, val_loss))

        if val_loss < best_loss:
            torch.save(net, args.model_save+'.'+args.dataset+'.'+args.save_flag+'.'+args.loss)
            best_loss = val_loss
            print("New best Model, saved.")

        if scheduler.step(val_loss):
            print("Early Stopping!")
            break

    print('test_loss, %.4f'%(best_loss))
