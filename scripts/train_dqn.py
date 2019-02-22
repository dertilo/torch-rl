import argparse

import torch

from envs import SnakeDQNAgent, build_SnakeEnv
from envs.cartpole import CartPoleAgent, build_CartPoleEnv
from scripts.visualize import visualize_it
from torch_rl.algos.dqn import DQNAlgo
from torch_rl.algos.train_methods import CsvLogger
from utils.general import set_seeds
from utils.save import get_logger, get_csv_writer, save_model

try:
    import gym_minigrid
except ImportError:
    raise Exception('gym_minigrid must be in PYTHONPATH!')

model_name = 'mlp-128-64'
args = argparse.Namespace(**{
    'model_name':model_name,
    'seed':1,
    'num_envs':1,
    'num_batches':10000,
    'num_rollout_steps':1

})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = 'storage/'+args.model_name

logger = get_logger(model_dir)
csv_file, csv_writer = get_csv_writer(model_dir)
set_seeds(args.seed)

# envs = build_CartPoleEnv(num_envs=args.num_envs, use_multiprocessing=False)
# q_model = CartPoleAgent(envs.observation_space, envs.action_space)
# target_model = CartPoleAgent(envs.observation_space, envs.action_space)

envs = build_SnakeEnv(num_envs=16, use_multiprocessing=True)
agent = SnakeDQNAgent(envs.observation_space, envs.action_space)
target_model = SnakeDQNAgent(envs.observation_space, envs.action_space)

logger.info("Model successfully created\n")
logger.info("{}\n".format(agent))

if torch.cuda.is_available():
    agent.cuda()
logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

algo = DQNAlgo(envs, agent, target_model, num_rollout_steps=args.num_rollout_steps,
               lr=0.0001,double_dpn=False,
               target_model_update_interval=20)
algo.train_model(args.num_batches,CsvLogger(csv_file,csv_writer,logger))
# env = build_CartPoleEnv(num_envs=1, use_multiprocessing=False)
# visualize_it(env,model_dir)
visualize_it(build_SnakeEnv(num_envs=1, use_multiprocessing=False),agent)