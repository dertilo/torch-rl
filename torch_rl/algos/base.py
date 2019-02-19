from abc import ABC, abstractmethod

import gym
import torch

from model import ACModel
from torch_rl.utils.dictlist import DictList

def flatten_arrays_in_dict(d):
    return {k: flatten_array(v) for k, v in d.items()}


def flatten_array(v):
    return v.transpose(0, 1).reshape(v.shape[0] * v.shape[1], *v.shape[2:])


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, env:gym.Env, acmodel:ACModel, num_rollout_steps, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, num_recurr_steps, reshape_reward):

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
        self.reshape_reward = reshape_reward

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

        # Initialize log values
        self.last_dones = torch.zeros(self.num_envs, device=self.device)
        self.rewards_sum = torch.zeros(self.num_envs, device=self.device) # TODO: wrap to be logged data in dict or something!
        self.log_episode_reshaped_return = torch.zeros(self.num_envs, device=self.device)
        self.num_steps_sum = torch.zeros(self.num_envs, device=self.device)

        self.log_done_counter = 0
        self.log_episode_rewards = []
        self.log_num_steps = []


    def collect_experiences(self):
        self.env_steps,self.agent_steps = self.gather_exp_via_rollout(self.env_steps,self.agent_steps)

        advantages = self.generalized_advantage_estimation(
            rewards=self.env_steps[1:].get('reward'),
            values=self.agent_steps.get('values'),
            dones=self.env_steps[1:].get('done'))

        exp = DictList(**{
            'env_steps':DictList(**flatten_arrays_in_dict(self.env_steps[:-1])),
            'agent_steps':DictList(**flatten_arrays_in_dict(self.agent_steps[:-1])),
            'advantages': flatten_array(advantages),
            'returnn':flatten_array(self.agent_steps[:-1].get('values')+advantages)
               })

        keep = max(self.log_done_counter, self.num_envs)# in one rollout there can be multiple dones!!

        log = {
            "return_per_episode": self.log_episode_rewards[-keep:],
            "num_frames_per_episode": self.log_num_steps[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_episode_rewards = self.log_episode_rewards[-self.num_envs:]
        self.log_num_steps = self.log_num_steps[-self.num_envs:]

        return exp, log

    def gather_exp_via_rollout(self, env_steps,agent_steps):
        env_steps[0] = env_steps[-1]
        agent_steps[0] = agent_steps[-1]
        self.acmodel.set_hidden_state(self.agent_steps[0])

        for i in range(self.num_rollout_steps):
            env_steps[i+1] = self.env.step(agent_steps[i])
            with torch.no_grad():
                agent_steps[i+1] = self.acmodel.step(env_steps[i+1])

            self.logging_stuff(env_steps.get('done')[i+1], env_steps.get('reward')[i+1])

        return env_steps,agent_steps

    def logging_stuff(self, done, reward):
        self.rewards_sum += torch.tensor(reward, device=self.device, dtype=torch.float)
        self.num_steps_sum += torch.ones(self.num_envs, device=self.device)
        for i, done_ in enumerate(done):
            if done_:
                self.log_done_counter += 1
                self.log_episode_rewards.append(self.rewards_sum[i].item())
                self.log_num_steps.append(self.num_steps_sum[i].item())
        self.rewards_sum *= 1 - done
        self.num_steps_sum *= 1 - done

    def generalized_advantage_estimation(self,rewards,values,dones):
        def calc_advantage(dones, rewards, values, num_rollout_steps, discount, gae_lambda):
            advantage_buffer = torch.zeros_like(rewards)
            next_advantage = 0
            for i in reversed(range(num_rollout_steps)):
                bellman_delta = rewards[i] + discount * values[i + 1] * (1 - dones[i]) - values[i]
                advantage_buffer[i] = bellman_delta + discount * gae_lambda * next_advantage * (1 - dones[i])
                next_advantage = advantage_buffer[i]
            return advantage_buffer
        # --------------------------------------------------------
        assert values.shape[0]==1+self.num_rollout_steps

        advantage_buffer = calc_advantage(dones, rewards, values,self.num_rollout_steps,self.discount,self.gae_lambda)
        return advantage_buffer


    @abstractmethod
    def update_parameters(self):
        pass
