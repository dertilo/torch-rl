from abc import ABC, abstractmethod
from collections import namedtuple

import gym
import torch
import numpy

from torch_rl.format import default_preprocess_obss
from torch_rl.utils import DictList, ParallelEnv

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, env:gym.Env, acmodel, num_rollout_steps, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

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
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_observation = self.env.reset()

        self.num_envs = len(self.last_observation)
        self.num_frames = self.num_rollout_steps * self.num_envs

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_rollout_steps % self.recurrence == 0

        # Initialize experience values

        shape = (self.num_rollout_steps, self.num_envs)


        self.obss = [None]*(shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_envs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_envs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_envs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_envs
        self.log_reshaped_return = [0] * self.num_envs
        self.log_num_frames = [0] * self.num_envs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_rollout_steps * num_envs, ...). k-th block
            of consecutive `self.num_rollout_steps` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        exp = {key:torch.zeros(*(self.num_rollout_steps, self.num_envs), device=self.device)
               for key in ['actions','values','logprobs','rewards','dones']}
        exp['observations'] = [None for _ in range(self.num_rollout_steps)]
        last_observation = self.last_observation

        for i in range(self.num_rollout_steps):
            preprocessed_obs = self.preprocess_obss(last_observation, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, values, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, values = self.acmodel(preprocessed_obs)
            action = dist.sample()
            logprob = dist.log_prob(action)
            obs, reward, done, _ = self.env.step(action.cpu().numpy())
            reward = torch.tensor(reward, device=self.device)
            done = torch.tensor(done, device=self.device, dtype=torch.float)
            for key,vals in zip(exp.keys(),[action,values,logprob,reward,done]):
                exp[key][i]=vals
            exp['observations'][i]=last_observation
            last_observation = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)

            self.logging_stuff(done, reward)

        self.last_observation = last_observation

        preprocessed_obs = self.preprocess_obss(self.last_observation, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        self.advantages = self.generalized_advantage_estimation(
            rewards=exp['rewards'],
            values=torch.cat([exp['values'], next_value.unsqueeze(0)], dim=0),
            dones=exp['dones'])

        exps = DictList()
        exps.obs = [exp['observations'][i][j]
                    for j in range(self.num_envs)
                    for i in range(self.num_rollout_steps)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = exp['actions'].transpose(0, 1).reshape(-1)
        exps.value = exp['values'].transpose(0, 1).reshape(-1)
        exps.reward = exp['rewards'].transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = exp['logprobs'].transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_envs)

        log = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_envs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_envs:]
        self.log_num_frames = self.log_num_frames[-self.num_envs:]

        return exps, log

    def logging_stuff(self, done, reward):
        self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
        self.log_episode_num_frames += torch.ones(self.num_envs, device=self.device)
        for i, done_ in enumerate(done):
            if done_:
                self.log_done_counter += 1
                self.log_return.append(self.log_episode_return[i].item())
                self.log_num_frames.append(self.log_episode_num_frames[i].item())
        self.log_episode_return *= self.mask
        self.log_episode_num_frames *= self.mask

    def generalized_advantage_estimation(self,rewards,values,dones):
        assert values.shape[0]==1+self.num_rollout_steps
        advantage_buffer = torch.zeros_like(rewards)
        next_advantage = 0
        for i in reversed(range(self.num_rollout_steps)):
            bellman_delta = rewards[i] + self.discount * values[i+1] * (1-dones[i]) - values[i]
            advantage_buffer[i] = bellman_delta + self.discount * self.gae_lambda * next_advantage * (1 - dones[i])
            next_advantage = advantage_buffer[i]
        return advantage_buffer

    @abstractmethod
    def update_parameters(self):
        pass
