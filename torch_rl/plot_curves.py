import gzip
import os
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import os

if __name__ == '__main__':
    cwd = os.getcwd()
    path = cwd+'/scripts/storage'
    # header = 'update,frames,FPS,duration,rreturn_mean,rreturn_std,rreturn_min,rreturn_max,num_frames_mean,num_frames_std,num_frames_min,num_frames_max,entropy,value,policy_loss,value_loss,grad_norm,return_mean,return_std,return_min,return_max'.split(',')
    # header = 'update,duration,episodes,rewards,ep-len'.split(',')
    with open(path+'/'+os.listdir(path)[0]+'/log.csv',mode='rt') as f:
        cols = f.readline()[:-1].split(',')
        header = [h for h in cols]
        idx = cols.index('update')
        dur_idx = header.index('duration')
        header.pop(dur_idx)
    figs_axes = [plt.subplots(1, 1) for _ in header]
    model_names = os.listdir(path)

    for dir_name in os.listdir(path):
        log_file = path + '/' + dir_name + '/log.csv'
        my_data = genfromtxt(log_file, delimiter=',', skip_header=1)
        for (fig,axes),param in zip(figs_axes,header):
            axes.plot(my_data[:,dur_idx],my_data[:,cols.index(param)])
            fig.suptitle(param)
            axes.legend(model_names)

    # for (fig,axes),param in zip(figs_axes,header):
    #     handles, _ = axes.get_legend_handles_labels()
    # axes.legend(handles, model_names)

    plt.show()
