import gzip
import os
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt


def read_lines(file, mode ='b', encoding ='utf-8', limit=np.Inf):
    counter = 0
    with gzip.open(file, mode='r'+mode) if file.endswith('.gz') else open(file,mode='r'+mode) as f:
        for line in f:
            counter+=1
            if counter>limit:
                break
            if mode == 'b':
                yield line.decode(encoding).replace('\n','')
            elif mode == 't':
                yield line.replace('\n','')

if __name__ == '__main__':
    path = '/home/tilo/code/RL/torch-rl/scripts/storage'
    header = 'update,frames,FPS,duration,rreturn_mean,rreturn_std,rreturn_min,rreturn_max,num_frames_mean,num_frames_std,num_frames_min,num_frames_max,entropy,value,policy_loss,value_loss,grad_norm,return_mean,return_std,return_min,return_max'.split(',')

    fig, axes = plt.subplots(1, 1)

    for dir_name in os.listdir(path):
        # lines_g = read_lines(path+'/'+dir_name+'/log.csv')
        # header = next(lines_g)
        # data = [d.split(',') for d in lines_g]
        my_data = genfromtxt(path+'/'+dir_name+'/log.csv', delimiter=',',skip_header=1)
        axes.plot(my_data[:,header.index('update')],my_data[:,header.index('rreturn_mean')])
    plt.legend([dir_name for dir_name in os.listdir(path)])
    plt.show()
