#!/usr/bin/env python3

import argparse
import gym
import time
import datetime

import numpy
import torch
import torch_rl
import sys

from agent_models import ACModel
from scripts.visualize import visualize_it
import envs
from torch_rl.algos.a2c import A2CAlgo
from torch_rl.utils.penv import ParallelEnv
from utils.format import preprocess_images
from utils.general import get_model_dir, set_seeds, calc_stats
from utils.save import get_logger, get_csv_writer, save_model

try:
    import gym_minigrid
except ImportError:
    raise Exception('gym_minigrid must be in PYTHONPATH!')


def train_model(num_batches, algo, logger, csv_writer, csv_file):

    num_steps = 0
    total_start_time = time.time()
    update = 0

    for k in range(num_batches):

        update_start_time = time.time()
        logs = algo.update_parameters()
        update_end_time = time.time()

        num_steps += logs["num_frames"]
        update += 1


        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = int(time.time() - total_start_time)
        return_per_episode = calc_stats(logs["return_per_episode"])
        num_frames_per_episode = calc_stats(logs["num_frames_per_episode"])

        log_stuff(csv_file, csv_writer, duration, fps, logger, logs, num_frames_per_episode, num_steps,
                  return_per_episode, update)


def log_stuff(csv_file, csv_writer, duration, fps, logger, logs, num_frames_per_episode, num_steps, return_per_episode,
              update):
    header = ["update", "frames", "FPS", "duration"]
    data = [update, num_steps, fps, duration]
    header += ["rreturn_" + key for key in return_per_episode.keys()]
    data += return_per_episode.values()
    header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
    data += num_frames_per_episode.values()
    header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
    data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
    logger.info(
        "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
            .format(*data))
    header += ["return_" + key for key in return_per_episode.keys()]
    data += return_per_episode.values()
    if num_steps == 0:
        csv_writer.writerow(header)
    csv_writer.writerow(data)
    csv_file.flush()


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--algo", default='a2c',
                    help="algorithm to use: a2c | ppo (REQUIRED)")
env_name = 'MiniGrid-Snake-v0'
# env_name = 'MiniGrid-Empty-8x8-v0'
model_name = 'mlp-128-64'
args = argparse.Namespace(**{
    'model_name':model_name,
    'env_name':env_name,
    'seed':1,
    'num_envs':16,
    'num_batches':600,
    'num_rollout_steps':5

})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PreprocessWrapper(gym.Env):
    def __init__(self,env:gym.Env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, act):
        actions = act.get('actions').cpu().numpy()
        env_step = self.env.step(actions)
        return self.process_env_step(env_step)

    def process_env_step(self, env_step):
        return {'reward': torch.tensor(env_step.get('reward')),
                'done': torch.tensor(env_step.get('done'), dtype=torch.float),
                'image': preprocess_images(env_step.get('observation'), device=device)}

    def reset(self):
        env_step = self.env.reset()
        return self.process_env_step(env_step)

    def render(self, mode='human'):
        return self.env.render(mode)

class DictEnvWrapper(gym.Env):
    def __init__(self,env:gym.Env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        obs,reward,done,_ = self.env.step(action)
        if done:
            obs = self.env.reset()
        return {'observation':obs['image'],'reward':reward,'done':done}

    def reset(self):
        obs = self.env.reset()
        return {'observation':obs['image'],'reward':0,'done':False}

    def render(self, mode='human'):
        return self.env.render(mode)

model_dir = 'storage/'+args.model_name

logger = get_logger(model_dir)
csv_file, csv_writer = get_csv_writer(model_dir)

set_seeds(args.seed)


def build_env_supplier(i):
    def env_supplier():
        env = gym.make(args.env_name)
        env.seed(args.seed + 10000 * i)
        env = DictEnvWrapper(env)
        return env
    return env_supplier


envs = PreprocessWrapper(ParallelEnv.build(build_env_supplier, args.num_envs))

acmodel = ACModel(envs.observation_space, envs.action_space)
logger.info("Model successfully created\n")
logger.info("{}\n".format(acmodel))

if torch.cuda.is_available():
    acmodel.cuda()
logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

algo = A2CAlgo(envs, acmodel,num_rollout_steps=args.num_rollout_steps, discount=0.99, lr=7e-4, gae_lambda=0.95,
                        entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, num_recurr_steps=1)

train_model(args.num_batches,algo,logger,csv_writer,csv_file)

if torch.cuda.is_available():
    acmodel.cpu()
save_model(acmodel, model_dir)
if torch.cuda.is_available():
    acmodel.cuda()

class SingleEnvWrapper(gym.Env):
    def __init__(self,env:gym.Env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        return {k:[v] for k,v in self.env.step(action).items()}

    def reset(self):
        return {k:[v] for k,v in self.env.reset().items()}

    def render(self, mode='human'):
        return self.env.render(mode)

env = PreprocessWrapper(SingleEnvWrapper(DictEnvWrapper(gym.make(args.env_name))))
visualize_it(env,model_dir)