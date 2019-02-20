from abc import ABC, abstractmethod
from typing import Dict

import gym
import torch

from agent_models import ACModel
from torch_rl.utils.dictlist import DictList

def flatten_arrays_in_dict(d):
    return {k: flatten_array(v) for k, v in d.items()}


def flatten_array(v):
    return v.transpose(0, 1).reshape(v.shape[0] * v.shape[1], *v.shape[2:])

def generalized_advantage_estimation(rewards,values,dones,num_rollout_steps,discount,gae_lambda):
    assert values.shape[0] == 1 + num_rollout_steps
    advantage_buffer = torch.zeros(rewards.shape[0]-1,rewards.shape[1])
    next_advantage = 0
    for i in reversed(range(num_rollout_steps)):
        bellman_delta = rewards[i+1] + discount * values[i + 1] * (1 - dones[i+1]) - values[i]
        advantage_buffer[i] = bellman_delta + discount * gae_lambda * next_advantage * (1 - dones[i+1])
        next_advantage = advantage_buffer[i]
    return advantage_buffer

def logging_stuff(logged:Dict, done, reward,num_envs,device):
    logged['rewards_sum'] += torch.tensor(reward, device=device, dtype=torch.float)
    logged['num_steps_sum'] += torch.ones(num_envs, device=device)
    for i, done_ in enumerate(done):
        if done_:
            logged['log_done_counter'] += 1
            logged['log_episode_rewards'].append(logged['rewards_sum'][i].item())
            logged['log_num_steps'].append(logged['num_steps_sum'][i].item())
    logged['rewards_sum'] *= 1 - done
    logged['num_steps_sum'] *= 1 - done
    return logged


class BaseAlgo(ABC):

    def __init__(self, env:gym.Env, acmodel:ACModel, num_rollout_steps, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, num_recurr_steps):

        self.env = env
        self.acmodel = acmodel
        self.acmodel.train()
        self.num_rollout_steps = num_rollout_steps
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.num_recurr_steps = num_recurr_steps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        initial_env_step = self.env.reset()
        self.num_envs = len(initial_env_step['reward'])

        def fill_with_zeros( num_rollout_steps, d):
            return DictList({k: torch.zeros(*(num_rollout_steps,) + v.shape) for k, v in d.items()})

        self.env_steps = fill_with_zeros(self.num_rollout_steps+1, initial_env_step)
        self.env_steps[-1] = initial_env_step
        with torch.no_grad():
            initial_agent_step = self.acmodel.step(initial_env_step)
        self.agent_steps = fill_with_zeros(self.num_rollout_steps + 1, initial_agent_step)
        self.agent_steps[-1] = initial_agent_step

        self.num_frames = self.num_rollout_steps * self.num_envs

        self.logged = {
            'rewards_sum':torch.zeros(self.num_envs, device=self.device),
            'num_steps_sum':torch.zeros(self.num_envs, device=self.device),
            'log_done_counter':0,
            'log_episode_rewards':[],
            'log_num_steps':[]
        }

    def collect_experiences(self):
        self.env_steps,self.agent_steps = self.gather_exp_via_rollout(self.env_steps,self.agent_steps)

        advantages = generalized_advantage_estimation(
            rewards=self.env_steps.get('reward'),
            values=self.agent_steps.get('values'),
            dones=self.env_steps.get('done'),
            num_rollout_steps=self.num_rollout_steps,discount=self.discount,gae_lambda=self.gae_lambda
        )

        exp = DictList(**{
            'env_steps':DictList(**flatten_arrays_in_dict(self.env_steps[:-1])),
            'agent_steps':DictList(**flatten_arrays_in_dict(self.agent_steps[:-1])),
            'advantages': flatten_array(advantages),
            'returnn':flatten_array(self.agent_steps[:-1].get('values')+advantages)
               })

        keep = max(self.logged['log_done_counter'], self.num_envs)# in one rollout there can be multiple dones!!

        log = {
            "return_per_episode": self.logged['log_episode_rewards'][-keep:],
            "num_frames_per_episode": self.logged['log_num_steps'][-keep:],
            "num_frames": self.num_frames
        }

        self.logged['log_done_counter'] = 0
        self.logged['log_episode_rewards'] = self.logged['log_episode_rewards'][-self.num_envs:]
        self.logged['log_num_steps'] = self.logged['log_num_steps'][-self.num_envs:]

        return exp, log

    def gather_exp_via_rollout(self, env_steps,agent_steps):
        env_steps[0] = env_steps[-1]
        agent_steps[0] = agent_steps[-1]
        self.acmodel.set_hidden_state(agent_steps[0])

        for i in range(self.num_rollout_steps):
            env_steps[i+1] = self.env.step(agent_steps[i])
            with torch.no_grad():
                agent_steps[i+1] = self.acmodel.step(env_steps[i+1])

            logging_stuff(self.logged,env_steps.get('done')[i+1], env_steps.get('reward')[i+1],self.num_envs,self.device)

        return env_steps,agent_steps




    @abstractmethod
    def update_parameters(self):
        pass
