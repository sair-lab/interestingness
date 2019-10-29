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

def detected(N, K, obj):
    window = np.sort(result[max(0,obj-N+1):obj+1,1])[::-1]
    if window.shape[0] < K or result[obj, 1] >= window[K-1]:
        return True
    else:
        return False


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser(description='Evaluate Interestingness')
    parser.add_argument("--source", type=str, default='results/source.txt', help="ground-truth file")
    parser.add_argument("--result", type=str, default='results/result.txt', help="results file")
    parser.add_argument("--min-object", type=int, default=10, help="minimum object number")
    parser.add_argument("--resolution", type=int, default=100, help="the number of different sliding windows")
    parser.add_argument('--nargs-int-type', nargs='+', type=int)
    parser.add_argument("--sigma", nargs='+', type=float, default=[1,2,3,5,10], help="top sigma*k interests")
    args = parser.parse_args(); print(args)
    
    source = np.loadtxt(args.source, dtype=int)
    result = np.loadtxt(args.result)
    
    objects = source.shape[0]
    length = result.shape[0]
    
    accuracy = np.ones(args.resolution+1)
    frames = np.zeros(length, dtype=int)
    frames[source] = 1

    for sigma in args.sigma:
        for idx in range(1, args.resolution+1):
            detect, N = 0, idx*length//args.resolution
            for obj in source:
                K = max(10, int(frames[max(0,obj-N+1):obj+1].sum()*sigma))
                if detected(N, K, obj) is True:
                    detect += 1
            accuracy[idx] = detect/objects

        x_axis, y_axis = np.array(range(args.resolution+1))/args.resolution, accuracy
        line, = plt.plot(x_axis, y_axis, label='Top K='+str(sigma))
        plt.legend()
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.gca().set_aspect("equal")
    plt.show()
