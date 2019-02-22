import time

import torch
from typing import Dict

from torch_rl.utils.dictlist import DictList
from utils.general import calc_stats


def flatten_parallel_rollout(d):
    return {k: flatten_parallel_rollout(v) if isinstance(v, dict) else flatten_array(v) for k, v in d.items()}


def flatten_array(v):
    return v.transpose(0, 1).reshape(v.shape[0] * v.shape[1], *v.shape[2:])

class CsvLogger(object):
    def __init__(self,csv_file,csv_writer,logger):
        self.num_steps = 0
        self.update = 0

        self.csv_file = csv_file
        self.csv_writer = csv_writer
        self.logger = logger

    def on_train_start(self):
        self.total_start_time = time.time()

    def log_it(self,logs,update_start_time,update_end_time):

        self.update += 1

        duration = int(time.time() - self.total_start_time)

        header = ["update", "duration"]
        logs = {k:v for k,v in zip(header+list(logs.keys()),[self.update,duration]+list(logs.values()))}
        s = ' '.join([k+'=%0.2f'%v for k,v in logs.items()])
        self.logger.info(s)

        if self.num_steps == 0:
            self.csv_writer.writerow(list(logs.keys()))
        self.csv_writer.writerow(list(logs.values()))
        self.csv_file.flush()

def log_step(log:Dict, exp:DictList):
    if log=={}:
        num_envs = len(exp)
        log_initial = {
            'rewards_sum':torch.zeros((num_envs,)),
            'num_steps_sum':torch.zeros((num_envs,)),
            'log_done_counter':0,
            'log_episode_rewards':[],
            'log_num_steps':[]
        }
        for k,v in log_initial.items():
            log[k]=v

    reward = exp.env.reward
    done = exp.env.done

    log['rewards_sum'] += torch.tensor(reward, dtype=torch.float)
    log['num_steps_sum'] += torch.ones(reward.shape[0])
    for i, done_ in enumerate(done):
        if done_:
            log['log_done_counter'] += 1
            log['log_episode_rewards'].append(log['rewards_sum'][i].item())
            log['log_num_steps'].append(log['num_steps_sum'][i].item())
    mask = torch.tensor(1 - done, dtype=torch.float)
    log['rewards_sum'] *= mask
    log['num_steps_sum'] *= mask


def fill_with_zeros(dim, d):
    return DictList(**{k: torch.zeros(*(dim,) + v.shape,dtype=v.dtype) if not isinstance(v, dict) else fill_with_zeros(dim, v) for k, v in d.items()})

class ExperienceMemory(object):

    def __init__(self, buffer_capacity: int,datum:DictList,logging_fun):

        self.buffer_capacity = buffer_capacity
        self.current_idx = 0
        self.last_written_idx = 0
        self.buffer = fill_with_zeros(buffer_capacity, datum)
        self.logging_fun = logging_fun
        self.log = {}

        self.store_single(datum)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value

    def inc_idx(self):
        self.last_written_idx = self.current_idx
        self.current_idx = (self.current_idx + 1) % self.buffer_capacity

    def store_single(self, datum:DictList):
        self.logging_fun(self.log, datum)
        self.buffer[self.current_idx]=datum
        self.inc_idx()
        return self.current_idx

    # def sample_batch(self,batch_size):
    #     indexes = torch.randint(0,10,(batch_size,))
    #     return self.buffer[indexes]
