import gzip
import os
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import os


def plot_curves(path,show=False):
    # header = 'update,frames,FPS,duration,rreturn_mean,rreturn_std,rreturn_min,rreturn_max,num_frames_mean,num_frames_std,num_frames_min,num_frames_max,entropy,value,policy_loss,value_loss,grad_norm,return_mean,return_std,return_min,return_max'.split(',')
    # header = 'update,duration,episodes,rewards,ep-len'.split(',')
    model_names = [d for d in os.listdir(path) if os.path.isdir(path+'/'+d)]
    with open(path + '/' + model_names[0] + '/log.csv', mode='rt') as f:
        cols = f.readline()[:-1].split(',')
        header = [h for h in cols]
        idx = cols.index('update')
        dur_idx = header.index('duration')
        header.pop(dur_idx)
    figs_axes = [plt.subplots(1, 1) for _ in header]
    for dir_name in model_names:
        log_file = path + '/' + dir_name + '/log.csv'
        my_data = genfromtxt(log_file, delimiter=',', skip_header=1)
        for (fig, axes), param in zip(figs_axes, header):
            axes.plot(my_data[:, dur_idx], my_data[:, cols.index(param)])
            fig.suptitle(param)
            axes.legend(model_names)
    for (fig, axes), param in zip(figs_axes, header):
        fig.savefig(path + '/' + param + '.png')
    # for (fig,axes),param in zip(figs_axes,header):
    #     handles, _ = axes.get_legend_handles_labels()
    # axes.legend(handles, model_names)
    if show:
        plt.show()


if __name__ == '__main__':
    plot_curves(os.getcwd()+'/storage')
