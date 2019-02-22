import argparse

import torch

from envs.cartpole import CartPoleAgent, build_CartPoleEnv
from torch_rl.algos.dqn import DQNAlgo
from torch_rl.algos.train_methods import CsvLogger
from utils.general import set_seeds
from utils.save import get_logger, get_csv_writer, save_model

try:
    import gym_minigrid
except ImportError:
    raise Exception('gym_minigrid must be in PYTHONPATH!')

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--algo", default='a2c',
                    help="algorithm to use: a2c | ppo (REQUIRED)")
# env_name = 'MiniGrid-Snake-v0'
# env_name = 'MiniGrid-Empty-8x8-v0'
model_name = 'mlp-128-64'
args = argparse.Namespace(**{
    'model_name':model_name,
    'env_name':'CartPole-v1',
    'seed':1,
    'num_envs':1,
    'num_batches':5000,
    'num_rollout_steps':1

})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = 'storage/'+args.model_name

logger = get_logger(model_dir)
csv_file, csv_writer = get_csv_writer(model_dir)

set_seeds(args.seed)

envs = build_CartPoleEnv(num_envs=args.num_envs, use_multiprocessing=False)


q_model = CartPoleAgent(envs.observation_space, envs.action_space)
target_model = CartPoleAgent(envs.observation_space, envs.action_space)
logger.info("Model successfully created\n")
logger.info("{}\n".format(q_model))

if torch.cuda.is_available():
    q_model.cuda()
logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

algo = DQNAlgo(envs, q_model,target_model, num_rollout_steps=args.num_rollout_steps,
               lr=0.001,
               target_model_update_interval=10)
algo.train_model(args.num_batches,CsvLogger(csv_file,csv_writer,logger))
#
if torch.cuda.is_available():
    q_model.cpu()
save_model(q_model, model_dir)
# if torch.cuda.is_available():
#     acmodel.cuda()
#
# env = build_CartPoleEnv(num_envs=1, use_multiprocessing=False)
# visualize_it(env,model_dir)