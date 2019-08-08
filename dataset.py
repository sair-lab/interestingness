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
import cv2
import glob
import torch
import argparse
import numpy as np
import torchvision
from PIL import Image
from random import sample
from operator import itemgetter
import torch.utils.data as Data
from torchvision import transforms, utils
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class VideoData(Dataset):
    def __init__(self, root, file, transform=None):
        self.frames = None
        cap = cv2.VideoCapture(os.path.join(root, file))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps =  int(cap.get(cv2.CAP_PROP_FPS))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames = torch.Tensor(nframes, 3, height, width)
        for i in range(nframes):
            _, frame = cap.read()
            frame = Image.fromarray(frame)
            if transform is not None:
                frame = transform(frame)
            self.frames[i,:,:,:] = frame

    def __len__(self):
        return self.frames.size(0)

    def __getitem__(self, idx):
        return self.frames[idx,:,:,:]


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
        return self.transform(image)


def save_batch(batch, folder, batch_idx):
    min_v = torch.min(batch)
    range_v = torch.max(batch) - min_v
    if range_v > 0:
        batch = (batch - min_v) / range_v
    else:
        batch = torch.zeros(batch.size())
    torchvision.utils.save_image(batch, folder+str(batch_idx)+'.png')


if __name__ == "__main__":
    from torch.autograd import Variable

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument("--data-root", type=str, default='.', help="dataset root folder")
    args = parser.parse_args(); print(args)

    transform = transforms.Compose([
        # transforms.RandomResizedCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # video = VideoData(root=args.data_root, file='data/train3.avi', transform=transform)
    # loader = Data.DataLoader(dataset=video, batch_size=1, shuffle=False)
    
    images = ImageData('dronefilm/unintrests', transform=transform)
    loader = Data.DataLoader(dataset=images, batch_size=1, shuffle=False)

    for batch_idx, frame in enumerate(loader):
        # if batch_idx%15==0:
            # save_batch(frame, 'data/unintrests/train3-', batch_idx)
        print(batch_idx)


