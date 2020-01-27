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
import glob
import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from evaluation import evaluate

# file0 = ['/data/datasets/SubTF/ground-truth/0817-ugv0-tunnel0-Hongbiao.txt',
#          '/data/datasets/SubTF/ground-truth/0817-ugv0-tunnel0-yuheng.txt',
#          "/data/datasets/SubTF/ground-truth/0817-ugv0-tunnel0-weikun.txt"]

# file1 = ["/data/datasets/SubTF/ground-truth/0817-ugv1-tunnel0-Hongbiao.txt",
#          "/data/datasets/SubTF/ground-truth/0817-ugv1-tunnel0-Huai.txt",
#          "/data/datasets/SubTF/ground-truth/0817-ugv1-tunnel0-yuheng.txt"]

# file2 = ['/data/datasets/SubTF/ground-truth/0818-ugv0-tunnel1-Hongbiao.txt',
#          "/data/datasets/SubTF/ground-truth/0818-ugv0-tunnel1-Junjun.txt",
#          '/data/datasets/SubTF/ground-truth/0818-ugv0-tunnel1-huai.txt']

# file3 = ["/data/datasets/SubTF/ground-truth/0818-ugv1-tunnel1-Hongbiao.txt",
#          "/data/datasets/SubTF/ground-truth/0818-ugv1-tunnel1-yaoyuh.txt",
#          "/data/datasets/SubTF/ground-truth/0818-ugv1-tunnel1-yuheng.txt"]

# file4 = ['/data/datasets/SubTF/ground-truth/0820-ugv0-tunnel1-Hongbiao.txt',
#          "/data/datasets/SubTF/ground-truth/0820-ugv0-tunnel1-shibo.txt",
#          "/data/datasets/SubTF/ground-truth/0820-ugv0-tunnel1-yaoyuh.txt"]

# file5 = ["/data/datasets/SubTF/ground-truth/0821-ugv0-tunnel0-Huai.txt",
#          "/data/datasets/SubTF/ground-truth/0821-ugv0-tunnel0-shibo.txt",
#          "/data/datasets/SubTF/ground-truth/0821-ugv0-tunnel0-yaoyuh.txt"]

# file6 = ["/data/datasets/SubTF/ground-truth/0821-ugv1-tunnel0-Hongbiao.txt",
#          "/data/datasets/SubTF/ground-truth/0821-ugv1-tunnel0-shibo.txt",
#          "/data/datasets/SubTF/ground-truth/0821-ugv1-tunnel0-yaoyuh.txt"]

# interest1 ground truth
file0 = ['/data/datasets/SubTF/ground-truth/0817-ugv0-tunnel0-interest-1.txt']
file1 = ["/data/datasets/SubTF/ground-truth/0817-ugv1-tunnel0-interest-1.txt"]
file2 = ['/data/datasets/SubTF/ground-truth/0818-ugv0-tunnel1-interest-1.txt']
file3 = ["/data/datasets/SubTF/ground-truth/0818-ugv1-tunnel1-interest-1.txt"]
file4 = ['/data/datasets/SubTF/ground-truth/0820-ugv0-tunnel1-interest-1.txt']
file5 = ["/data/datasets/SubTF/ground-truth/0821-ugv0-tunnel0-interest-1.txt"]
file6 = ["/data/datasets/SubTF/ground-truth/0821-ugv1-tunnel0-interest-1.txt"]
interest1 = [file0, file1, file2, file3, file4, file5, file6]

# interest2 ground truth
file0 = ['/data/datasets/SubTF/ground-truth/0817-ugv0-tunnel0-interest-2.txt']
file1 = ["/data/datasets/SubTF/ground-truth/0817-ugv1-tunnel0-interest-2.txt"]
file2 = ['/data/datasets/SubTF/ground-truth/0818-ugv0-tunnel1-interest-2.txt']
file3 = ["/data/datasets/SubTF/ground-truth/0818-ugv1-tunnel1-interest-2.txt"]
file4 = ['/data/datasets/SubTF/ground-truth/0820-ugv0-tunnel1-interest-2.txt']
file5 = ["/data/datasets/SubTF/ground-truth/0821-ugv0-tunnel0-interest-2.txt"]
file6 = ["/data/datasets/SubTF/ground-truth/0821-ugv1-tunnel0-interest-2.txt"]
interest2 = [file0, file1, file2, file3, file4, file5, file6]

# from personalvideo import *

if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Evaluate Interestingness')
    parser.add_argument("--dataset", type=str, default='SubTF', help="file save flag name")
    parser.add_argument("--save-flag", type=str, default='r5w5', help="file save flag name")
    parser.add_argument('--root', type=str, default='results', help='results folder')
    parser.add_argument("--min-object", type=int, default=1, help="minimum number of top interests")
    parser.add_argument("--resolution", type=int, default=100, help="number of points of the plotted lines")
    parser.add_argument("--tol", type=int, default=1, help="the maximum tolerant frames")
    parser.add_argument("--category", type=str, default='interest-1', help="interest-1 or interest-2")
    parser.add_argument("--delta", nargs='+', type=float, default=[1,2,4], help="top delta*K are accepted, where K is truth")
    args = parser.parse_args(); print(args)

    if args.category is 'interest-1':
        sources = interest1
    elif args.category is 'interest-2':
        sources = interest2
    else:
        NotImplementedError('please select interest-1 or interest-2')

    mean_accuracies = np.zeros((len(args.delta), args.resolution+1))
    mean_means = 0 
    x = np.array(range(args.resolution+1))/args.resolution
    for test_id in range(len(sources)):
        target = glob.glob(os.path.join(args.root, args.dataset+'-%d-*-%s.txt'%(test_id, args.save_flag)))
        accuracies = np.zeros((len(args.delta), args.resolution+1))
        means = np.zeros(len(args.delta))
        for source in sources[test_id]:
            accuracy, mean = evaluate(source, target[0], args.min_object, args.resolution, args.tol, args.delta)
            accuracies += accuracy
            means += mean
        
        num = len(sources[test_id])
        accuracies = accuracies/num
        means = means/num
        figure(num=test_id, figsize=(4, 4), facecolor='w', edgecolor='k')
        
        mean_accuracies += accuracies
        mean_means += means

        for i in range(len(args.delta)):
            line, = plt.plot(x, accuracies[i,:], label='[%.2f'%(means[i])+r'] $\delta$='+str(args.delta[i]))
            plt.legend()
            plt.grid()
            plt.xlim((0,1))
            plt.ylim((0,1))
            plt.gca().set_aspect("equal")
    mean_accuracies /= len(sources)
    mean_means /= len(sources)
    print("mean accuracy:", mean_means)
    np.savetxt('performance/%s-accuracy-tol=%d-'%(args.category, args.tol)+args.save_flag+'.txt', mean_accuracies.transpose(),fmt='%.6f')

    figure(num=100, figsize=(4, 4), facecolor='w', edgecolor='k')
    for i in range(len(args.delta)):
        line, = plt.plot(x, mean_accuracies[i,:], label='[%.2f'%(mean_means[i])+r'] $\delta$='+str(args.delta[i]))
        plt.legend()
        plt.grid()
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.gca().set_aspect("equal")

    plt.title(r'Accuracy ($\tau$=%d)'%(args.tol))
    plt.xlabel('sequence length')
    plt.ylabel('accuracy')
    plt.show()
