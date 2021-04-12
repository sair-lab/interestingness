#!/usr/bin/env python3

import os
import copy
import tqdm
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.utils.data as Data
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection

from autoencoder import AutoEncoder
from torchutil import EarlyStopScheduler, count_parameters


def train(loader, net, creterion):
    train_loss, batches = 0, len(loader)
    enumerater = tqdm.tqdm(enumerate(loader))
    for batch_idx, (inputs, _) in enumerater:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = creterion(inputs, outputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        enumerater.set_description("train loss: %.4f on %d/%d"%(train_loss/(batch_idx+1), batch_idx, batches))
    return train_loss/(batch_idx+1)


def performance(loader, net, creterion):
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = net(inputs)
            loss = creterion(inputs, outputs)
            test_loss += loss.item()
    return test_loss/(batch_idx+1)


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Train AutoEncoder')
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset root folder")
    parser.add_argument("--model", type=str, default='vgg', help="vgg, resnet, or mobilenet")
    parser.add_argument('--crop-size', nargs='+', type=int, default=[320,320], help='image crop size')
    parser.add_argument("--model-save", type=str, default='saves/vgg16.pt', help="model save point")
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--factor", type=float, default=0.1, help="ReduceLROnPlateau factor")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="minimum lr for ReduceLROnPlateau")
    parser.add_argument("--patience", type=int, default=5, help="patience of epochs for ReduceLROnPlateau")
    parser.add_argument("--epochs", type=int, default=150, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="number of minibatch size")
    parser.add_argument("--momentum", type=float, default=0, help="momentum of the optimizer")
    parser.add_argument("--alpha", type=float, default=0.1, help="weight of TVLoss")
    parser.add_argument("--w-decay", type=float, default=1e-5, help="weight decay of the optimizer")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers for dataloader")
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.set_defaults(self_loop=False)
    args = parser.parse_args(); print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.makedirs("saves", exist_ok=True)
    with open(args.model_save+'.txt','a+') as f:
        f.write(str(args)+'\n')

    train_transform = transforms.Compose([
            # transforms.RandomRotation(20),
            transforms.RandomResizedCrop(tuple(args.crop_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_transform = transforms.Compose([
            transforms.CenterCrop(tuple(args.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_root = os.path.join(args.data_root, 'coco/images/train2017')
    val_root = os.path.join(args.data_root, 'coco/images/val2017')
    test_root = os.path.join(args.data_root, 'coco/images/test2017')

    train_annFile = os.path.join(args.data_root, 'coco/annotations/annotations_trainval2017/captions_train2017.json')
    val_annFile = os.path.join(args.data_root, 'coco/annotations/annotations_trainval2017/captions_val2017.json')
    test_annFile = os.path.join(args.data_root, 'coco/annotations/image_info_test2017/image_info_test2017.json')

    train_data = CocoDetection(root=train_root, annFile=train_annFile, transform=train_transform)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    val_data = CocoDetection(root=val_root, annFile=val_annFile, transform=val_transform)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    if args.resume == True:
        net, best_loss = torch.load(args.model_save)
        print("Resume train from {} with loss {}".format(args.model_save, best_loss))
    else:
        net = AutoEncoder(args.model)
        best_loss = float('Inf')

    if torch.cuda.is_available():
        print("Runnin on {} GPU".format(list(range(torch.cuda.device_count()))))
        net = nn.DataParallel(net.cuda(), device_ids=list(range(torch.cuda.device_count())))

    creterion = nn.MSELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.w_decay)
    scheduler = EarlyStopScheduler(optimizer, factor=args.factor, verbose=True, min_lr=args.min_lr, patience=args.patience)

    print('number of parameters:', count_parameters(net))
    for epoch in range(args.epochs):
        train_loss = train(train_loader, net, creterion)
        val_loss = performance(val_loader, net, creterion) # validate

        with open(args.model_save+'.txt','a+') as f:
            f.write("epoch: %d, train_loss: %.4f, val_loss: %.4f, lr: %f\n" % (epoch, train_loss, val_loss, optimizer.param_groups[0]['lr']))

        if val_loss < best_loss:
            print("New best Model, saving...")
            torch.save((net.module, val_loss), args.model_save)
            best_loss = val_loss

        if scheduler.step(val_loss):
            print('Early Stopping!')
            break

    print("Testing")
    net, _ = torch.load(args.model_save)
    if torch.cuda.is_available():
        net = nn.DataParallel(net.cuda(), device_ids=list(range(torch.cuda.device_count())))

    test_data = CocoDetection(root=test_root, annFile=test_annFile, transform=val_transform)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    test_loss = performance(test_loader, net, creterion)
    print('val_loss: %.2f, test_loss, %.4f'%(best_loss, test_loss))
