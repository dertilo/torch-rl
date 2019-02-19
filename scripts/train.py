#!/usr/bin/env python3

import argparse
import gym
import time
import datetime
import torch
import torch_rl
import sys

from scripts.train_methods import train_model
from scripts.visualize import visualize_it
import envs
from torch_rl.algos.a2c import A2CAlgo
from torch_rl.algos.ppo import PPOAlgo
from torch_rl.utils.penv import ParallelEnv
from utils.format import preprocess_images
from utils.general import get_model_dir, set_seeds
from utils.save import get_logger, get_csv_writer, save_model

try:
    import gym_minigrid
except ImportError:
    raise Exception('gym_minigrid must be in PYTHONPATH!')

import utils
from model import ACModel

# Parse arguments

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--algo", default='a2c',
                    help="algorithm to use: a2c | ppo (REQUIRED)")
# env_name = 'MiniGrid-Snake-v0'
env_name = 'MiniGrid-Empty-8x8-v0'
model_name = 'mlp-128-64'
parser.add_argument("--env", default=env_name,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=model_name,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=80*300,
                    help="number of frames of training (default: 10e7)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 0, 0 means no saving)")
parser.add_argument("--tb", action="store_true", default=False,
                    help="log into Tensorboard")
parser.add_argument("--num-rollout-steps", type=int, default=5,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate for optimizers (default: 7e-4)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer apha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of timesteps gradient is backpropagated (default: 1)\nIf > 1, a LSTM is added to the model to have memory")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
args = parser.parse_args()
args.mem = args.recurrence > 1

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


# Define run dir

suffix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = "{}_{}_seed{}_{}".format(args.env, args.algo, args.seed, suffix)
model_name = args.model or default_model_name
model_dir = get_model_dir(model_name)

# Define logger, CSV writer and Tensorboard writer

logger = get_logger(model_dir)
csv_file, csv_writer = get_csv_writer(model_dir)
if args.tb:
    from tensorboardX import SummaryWriter
    tb_writer = SummaryWriter(model_dir)

# Log command and all script arguments

logger.info("{}\n".format(" ".join(sys.argv)))
logger.info("{}\n".format(args))

# Set seed for all randomness sources

set_seeds(args.seed)


def build_env_supplier(i):
    def env_supplier():
        env = gym.make(args.env)
        env.seed(args.seed + 10000 * i)
        env = DictEnvWrapper(env)
        return env
    return env_supplier


# Generate environments
envs = PreprocessWrapper(ParallelEnv.build(build_env_supplier, args.procs))
# Load training status

# try:
#     status = utils.load_status(model_dir)
# except OSError:

# Define actor-critic model

# try:
#     acmodel = utils.load_model(model_dir)
#     logger.info("Model successfully loaded\n")
# except OSError:

acmodel = ACModel(envs.observation_space, envs.action_space, args.mem, args.text)
logger.info("Model successfully created\n")
logger.info("{}\n".format(acmodel))

if torch.cuda.is_available():
    acmodel.cuda()
logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

# Define actor-critic algo

if args.algo == "a2c":
    algo = A2CAlgo(envs, acmodel, args.num_rollout_steps, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps)
elif args.algo == "ppo":
    algo = PPOAlgo(envs, acmodel, args.num_rollout_steps, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))


train_model(args.frames,algo,logger,csv_writer,csv_file)

# if torch.cuda.is_available():
#     acmodel.cpu()
# save_model(acmodel, model_dir)
# if torch.cuda.is_available():
#     acmodel.cuda()


# visualize_it(gym.make(args.env),model_dir)