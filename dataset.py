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
import glob
import glob2
import torch
import argparse
import numpy as np
import torchvision
from PIL import Image
from random import sample
from operator import itemgetter
import torch.utils.data as Data
from matplotlib import pyplot as plt
from torchvision import transforms, utils
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class VideoData(Dataset):
    def __init__(self, root, file, transform=None):
        self.transform = transform
        self.cap = cv2.VideoCapture(os.path.join(root, file))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps =  int(self.cap.get(cv2.CAP_PROP_FPS))
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def size(self):
        return (self.nframes, 3, self.height, self.width)

    def __len__(self):
        return self.nframes

    def __getitem__(self, idx):
        _, frame = self.cap.read()
        frame = Image.fromarray(frame)
        if self.transform is not None:
            frame = self.transform(frame)
        return frame


class ImageData(Dataset):
    def __init__(self, root, train=True, ratio=0.8, transform=None):
        self.transform = transform
        self.filename = []
        types = ('*.jpg','*.jpeg','*.png','*.ppm','*.bmp','*.pgm','*.tif','*.tiff','*.webp')
        for files in types:
            self.filename.extend(glob.glob(os.path.join(root, files)))

        indexfile = os.path.join(root, 'split.pt')
        N = len(self.filename)
        if os.path.exists(indexfile):
            train_index, test_index = torch.load(indexfile)
            assert len(train_index)+len(test_index) == N, 'Data changed! Pleate delete '+indexfile 
        else:
            indices = range(N)
            train_index = sample(indices, int(ratio*N))
            test_index = np.delete(indices, train_index)
            torch.save((train_index, test_index), indexfile)
        
        if train == True:
            self.filename = itemgetter(*train_index)(self.filename)
        else:
            self.filename = itemgetter(*test_index)(self.filename)

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, idx):
        image = Image.open(self.filename[idx])
        return self.transform(image), torch.tensor([])


class Dronefilm(Dataset):
    def __init__(self, root, data='car', test_id=0, train=True, transform=None):
        self.transform, self.train = transform, train

        if train is True:
            self.filenames = sorted(glob.glob(os.path.join(root, 'dronefilm', data, 'train/*.png')))
            self.nframes = len(self.filenames)
        else:
            filenames = sorted(glob.glob(os.path.join(root, 'dronefilm', data, 'test/*.avi')))
            cap = cv2.VideoCapture(filenames[test_id])
            print("Using test sequences:", filenames[test_id])
            self.nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frames = []
            for _ in range(self.nframes):
                _, frame = cap.read()
                frame = Image.fromarray(frame)
                self.frames.append(frame)

    def __len__(self):
        return self.nframes

    def __getitem__(self, idx):
        if self.train is True:
            frame = Image.open(self.filenames[idx])
        else:
            frame = self.frames[idx]
        if self.transform is not None:
            frame = self.transform(frame)
        return frame


def save_batch(batch, folder, batch_idx):
    torchvision.utils.save_image(batch, folder+"%04d"%batch_idx+'.png')


def show_batch(batch, name="video"):
    min_v = torch.min(batch)
    range_v = torch.max(batch) - min_v
    if range_v > 0:
        batch = (batch - min_v) / range_v
    else:
        batch = torch.zeros(batch.size())
    grid = torchvision.utils.make_grid(batch)
    img = grid.numpy()[::-1].transpose((1, 2, 0))
    cv2.imshow(name, img)
    cv2.waitKey(1)


if __name__ == "__main__":
    from torch.autograd import Variable

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument("--data-root", type=str, default='.', help="dataset root folder")
    args = parser.parse_args(); print(args)

    transform = transforms.Compose([
        transforms.Resize(384),
        # transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    video = VideoData(root='/data/datasets/subte', file='movie.mpg', transform=transform)
    loader = Data.DataLoader(dataset=video, batch_size=1, shuffle=False)
    
    # images = ImageData('dronefilm/unintrests', transform=transform)
    # loader = Data.DataLoader(dataset=images, batch_size=1, shuffle=False)

    # images = Mavscout('/data/datasets', transform=transform)
    # loader = Data.DataLoader(dataset=images, batch_size=1, shuffle=False)

    # data = Dronefilm(root="/data/datasets", data='car', test_id=0, train=False, transform=transform)
    # loader = Data.DataLoader(dataset=data, batch_size=1, shuffle=False)

    for batch_idx, frame in enumerate(loader):
        if batch_idx%15==0:
            # save_batch(frame, '/data/datasets/dronefilm/bike/train/t1-', batch_idx)
            show_batch(frame)
        # if batch_idx%15==0:
            # save_batch(frame, '/data/datasets/dronefilm/bike/train/t1-', batch_idx)
            print(batch_idx)


