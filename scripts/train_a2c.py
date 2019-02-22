import argparse
import time

import gym
import torch

from envs import build_SnakeEnv, SnakeAgent
from scripts.visualize import visualize_it
from torch_rl.algos.a2c import A2CAlgo
from torch_rl.algos.train_methods import CsvLogger
from utils.general import set_seeds, calc_stats
from utils.save import get_logger, get_csv_writer, save_model

try:
    import gym_minigrid
except ImportError:
    raise Exception('gym_minigrid must be in PYTHONPATH!')

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

model_dir = 'storage/'+args.model_name

logger = get_logger(model_dir)
csv_file, csv_writer = get_csv_writer(model_dir)

set_seeds(args.seed)

envs = build_SnakeEnv(num_envs=16, use_multiprocessing=True)
agent = SnakeAgent(envs.observation_space, envs.action_space)
logger.info("Model successfully created\n")
logger.info("{}\n".format(agent))

if torch.cuda.is_available():
    agent.cuda()
logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

algo = A2CAlgo(envs, agent, num_rollout_steps=args.num_rollout_steps, discount=0.99, lr=7e-4, gae_lambda=0.95,
               entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, num_recurr_steps=1)

algo.train_model(args.num_batches,CsvLogger(csv_file,csv_writer,logger))

visualize_it(build_SnakeEnv(num_envs=1, use_multiprocessing=False),agent)