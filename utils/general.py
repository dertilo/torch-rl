import random
import os
import numpy
import torch
import collections

def get_storage_dir():
    if "TORCH_RL_STORAGE" in os.environ:
        return os.environ["TORCH_RL_STORAGE"]
    return "storage"

def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

def set_seeds(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calc_stats(array):
    if len(array)>0:
        stats = {
            'mean': numpy.mean(array),
            'median': numpy.median(array),
            'std': numpy.std(array),
            'min': numpy.amin(array),
            'max': numpy.amax(array)
        }
    else:
        stats = {
            'mean': numpy.NaN,
            'median': numpy.NaN,
            'std': numpy.NaN,
            'min': numpy.NaN,
            'max': numpy.NaN
        }

    return stats