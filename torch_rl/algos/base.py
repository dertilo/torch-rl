from abc import ABC, abstractmethod
from collections import namedtuple

import gym
import torch
import numpy

from model import ACModel
from torch_rl.format import default_preprocess_obss
from torch_rl.utils.dictlist import DictList


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, env:gym.Env, acmodel:ACModel, num_rollout_steps, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, num_recurr_steps, preprocess_obss, reshape_reward):
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
        self.num_recurr_steps = num_recurr_steps
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.last_observation = self.env.reset()

        self.num_envs = len(self.last_observation)
        self.num_frames = self.num_rollout_steps * self.num_envs

        self.hidden_states = torch.zeros(self.num_envs, self.acmodel.hiddenstate_size, device=self.device)
        # Initialize log values
        self.last_dones = torch.zeros(self.num_envs, device=self.device)
        self.rewards_sum = torch.zeros(self.num_envs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_envs, device=self.device)
        self.num_steps_sum = torch.zeros(self.num_envs, device=self.device)

        self.log_done_counter = 0
        self.log_episode_rewards = []
        self.log_num_steps = []

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
        exp, self.last_observation = self.gather_exp_via_rollout(self.hidden_states, self.last_observation)

        self.hidden_states = exp['hidden_states'][-1]


        advantages = self.generalized_advantage_estimation(
            rewards=exp['rewards'],
            values=exp['values'],
            dones=exp['dones'])

        exps = self.repacking_experiences(advantages, exp)
        self.last_dones = exp['dones'][-1]
        keep = max(self.log_done_counter, self.num_envs)# in one rollout there can be multiple dones!!

        log = {
            "return_per_episode": self.log_episode_rewards[-keep:],
            "num_frames_per_episode": self.log_num_steps[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_episode_rewards = self.log_episode_rewards[-self.num_envs:]
        self.log_num_steps = self.log_num_steps[-self.num_envs:]

        return exps, log

    def repacking_experiences(self, advantages, exp):
        exps = DictList()
        exps.obs = [exp['observations'][i][j]
                    for j in range(self.num_envs)
                    for i in range(self.num_rollout_steps)]
        exps.memory = exp['hidden_states'].transpose(0, 1).reshape(-1, *exp['hidden_states'].shape[2:])
        mask = torch.cat((1 - self.last_dones.unsqueeze(0), 1 - exp['dones'][:-1]), dim=0)
        exps.mask = mask.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = exp['actions'].transpose(0, 1).reshape(-1)
        exps.value = exp['values'].transpose(0, 1).reshape(-1)
        exps.reward = exp['rewards'].transpose(0, 1).reshape(-1)
        exps.advantage = advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = exp['logprobs'].transpose(0, 1).reshape(-1)
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        return exps

    def gather_exp_via_rollout(self, hidden_states, last_observation):
        exp = {key: torch.zeros(*(self.num_rollout_steps, self.num_envs), device=self.device)
               for key in ['actions', 'values', 'logprobs', 'rewards', 'dones']}
        exp['hidden_states'] = torch.zeros(*(self.num_rollout_steps,) + tuple(self.hidden_states.shape),
                                           device=self.device)
        exp['infos'] = [None for _ in range(self.num_rollout_steps)]
        exp['observations'] = [None for _ in range(self.num_rollout_steps)]
        for i in range(self.num_rollout_steps):
            with torch.no_grad():
                dist, values, hidden_states = self.acmodel(self.preprocess_obss(last_observation, device=self.device),
                                                           hidden_states)
            action = dist.sample()
            logprob = dist.log_prob(action)
            obs, reward, dones, infos = self.env.step(action.cpu().numpy())
            hidden_states = hidden_states * (1 - dones).unsqueeze(1)

            for key, vals in zip(exp.keys(),
                                 [action, values, logprob, reward, dones, hidden_states, infos, last_observation]):
                exp[key][i] = vals
            last_observation = obs
            self.logging_stuff(dones, reward)
        return exp, last_observation

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
        with torch.no_grad():
            _, next_value, _ = self.acmodel(self.preprocess_obss(self.last_observation, device=self.device),
                                            self.hidden_states)
        values = torch.cat([values, next_value.unsqueeze(0)], dim=0)
        assert values.shape[0]==1+self.num_rollout_steps

        advantage_buffer = calc_advantage(dones, rewards, values,self.num_rollout_steps,self.discount,self.gae_lambda)
        return advantage_buffer


    @abstractmethod
    def update_parameters(self):
        pass
