#!/usr/bin/env python3

import os
import glob
import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from evaluation import evaluate


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Evaluate Interestingness')
    parser.add_argument("--data-root", type=str, default='/data/datasets', help="dataset root folder")
    parser.add_argument("--dataset", type=str, default='SubTF', help="file save flag name")
    parser.add_argument("--save-flag", type=str, default='n100usage', help="file save flag name")
    parser.add_argument('--root', type=str, default='results', help='results folder')
    parser.add_argument("--min-object", type=int, default=1, help="minimum number of top interests")
    parser.add_argument("--resolution", type=int, default=100, help="number of points of the plotted lines")
    parser.add_argument("--tol", type=int, default=1, help="the maximum tolerant frames")
    parser.add_argument("--category", type=str, default='normal', help="normal or difficult")
    parser.add_argument("--delta", nargs='+', type=float, default=[1,2,4], help="top delta*K are accepted, where K is truth")
    args = parser.parse_args(); print(args)

    # interest1 ground truth
    file0 = [args.data_root+"/SubTF/ground-truth/0817-ugv0-tunnel0-interest-1.txt"]
    file1 = [args.data_root+"/SubTF/ground-truth/0817-ugv1-tunnel0-interest-1.txt"]
    file2 = [args.data_root+"/SubTF/ground-truth/0818-ugv0-tunnel1-interest-1.txt"]
    file3 = [args.data_root+"/SubTF/ground-truth/0818-ugv1-tunnel1-interest-1.txt"]
    file4 = [args.data_root+"/SubTF/ground-truth/0820-ugv0-tunnel1-interest-1.txt"]
    file5 = [args.data_root+"/SubTF/ground-truth/0821-ugv0-tunnel0-interest-1.txt"]
    file6 = [args.data_root+"/SubTF/ground-truth/0821-ugv1-tunnel0-interest-1.txt"]
    interest1 = [file0, file1, file2, file3, file4, file5, file6]

    # interest2 ground truth
    file0 = [args.data_root+'/SubTF/ground-truth/0817-ugv0-tunnel0-interest-2.txt']
    file1 = [args.data_root+"/SubTF/ground-truth/0817-ugv1-tunnel0-interest-2.txt"]
    file2 = [args.data_root+"/SubTF/ground-truth/0818-ugv0-tunnel1-interest-2.txt"]
    file3 = [args.data_root+"/SubTF/ground-truth/0818-ugv1-tunnel1-interest-2.txt"]
    file4 = [args.data_root+"/SubTF/ground-truth/0820-ugv0-tunnel1-interest-2.txt"]
    file5 = [args.data_root+"/SubTF/ground-truth/0821-ugv0-tunnel0-interest-2.txt"]
    file6 = [args.data_root+"/SubTF/ground-truth/0821-ugv1-tunnel0-interest-2.txt"]
    interest2 = [file0, file1, file2, file3, file4, file5, file6]

    os.makedirs("performance", exist_ok=True)

    if args.category == 'normal':
        sources = interest1
    elif args.category == 'difficult':
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

    plt.title(r'Curve of Real-time Precision ($\tau$=%d)'%(args.tol))
    plt.xlabel('sequence length')
    plt.ylabel('accuracy')
    plt.show()
