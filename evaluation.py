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
import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def detected(result, N, K, obj, length, tol):
    window = np.sort(result[max(0,obj-N+1):obj+1,1])[::-1]
    if window.shape[0] < K or (result[max(0,obj-tol):min(obj+tol+1, length), 1] >= window[K-1]).sum()>0:
        return True
    return False


def evaluate(source, target, min_object=10, resolution=100, tol=2, delta=[1,2,4]):
    source = np.loadtxt(source, dtype=int)
    result = np.loadtxt(target)

    objects = source.shape[0]
    length = result.shape[0]

    frames = np.zeros(length, dtype=int)
    frames[source] = 1

    accuracies = []
    for delta in delta:
        accuracy = np.ones(resolution+1)
        for idx in range(1, resolution+1):
            detect, N = 0, idx*length//resolution
            for obj in source:
                K = max(min_object, int(frames[max(0,obj-N+1):obj+1].sum()*delta))
                if detected(result, N, K, obj, length, tol) is True:
                    detect += 1
            accuracy[idx] = detect/objects
        accuracies.append(accuracy)
    accuracies = np.stack(accuracies, axis=0)

    return accuracies, accuracies.mean(axis=1)


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Evaluate Interestingness')
    parser.add_argument("--source", type=str, help="ground-truth file")
    parser.add_argument("--target", type=str, help="results file")
    parser.add_argument("--min-object", type=int, default=10, help="minimum number of top interests")
    parser.add_argument("--resolution", type=int, default=100, help="number of points of the plotted lines")
    parser.add_argument("--tol", type=int, default=1, help="the maximum tolerant frames")
    parser.add_argument("--delta", nargs='+', type=float, default=[1,2,4], help="top delta*K are accepted, where K is truth")
    args = parser.parse_args(); print(args)
    
    x = np.array(range(args.resolution+1))/args.resolution
    accuracies, mean = evaluate(args.source, args.target, args.min_object, args.resolution, args.tol, args.delta)
    figure(num=1, figsize=(4, 4), facecolor='w', edgecolor='k')
    for i in range(accuracies.shape[0]):
        line, = plt.plot(x, accuracies[i,:], label='[%.2f'%(mean[i])+r'] $\delta$='+str(args.delta[i]))
        plt.legend()
        plt.grid()
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.gca().set_aspect("equal")
    print("Accuracy:",mean)
    plt.title(r'Accuracy ($\tau$=%d)'%(args.tol))
    plt.xlabel('sequence length')
    plt.ylabel('accuracy')
    plt.show()
