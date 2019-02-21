import argparse
import time

import gym
import torch

from agent_models import ACModel, QModel
from scripts.visualize import visualize_it
from torch_rl.algos.a2c import A2CAlgo
from torch_rl.algos.dqn import DQNAlgo
from torch_rl.algos.train_methods import CsvLogger
from torch_rl.utils.penv import ParallelEnv
from utils.format import preprocess_images
from utils.general import set_seeds, calc_stats
from utils.save import get_logger, get_csv_writer, save_model

try:
    import gym_minigrid
except ImportError:
    raise Exception('gym_minigrid must be in PYTHONPATH!')

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
    'num_batches':1000,
    'num_rollout_steps':10

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
        return {'reward': torch.tensor(env_step.get('reward'),dtype=torch.float),
                'done': torch.tensor(env_step.get('done')),
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

q_model = QModel(envs.observation_space, envs.action_space)
target_model = QModel(envs.observation_space, envs.action_space)
logger.info("Model successfully created\n")
logger.info("{}\n".format(q_model))

if torch.cuda.is_available():
    q_model.cuda()
logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

algo = DQNAlgo(envs, q_model,target_model, num_rollout_steps=args.num_rollout_steps,target_model_update_interval=10)
algo.train_model(args.num_batches,CsvLogger(csv_file,csv_writer,logger))
#
if torch.cuda.is_available():
    q_model.cpu()
save_model(q_model, model_dir)
# if torch.cuda.is_available():
#     acmodel.cuda()
#
env = PreprocessWrapper(SingleEnvWrapper(DictEnvWrapper(gym.make(args.env_name))))
visualize_it(env,model_dir)