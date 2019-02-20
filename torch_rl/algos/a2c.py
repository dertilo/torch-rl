from typing import Dict

import numpy
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


class A2CAlgo(object):
    """The class for the Advantage Actor-Critic algorithm."""

    def __init__(self, env, acmodel:ACModel, num_rollout_steps=None, discount=0.99, lr=7e-4, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, num_recurr_steps=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-5):

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

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        inds = numpy.arange(0, self.num_frames, self.num_recurr_steps)
        self.acmodel.set_hidden_state(exps[inds].agent_steps)
        for i in range(self.num_recurr_steps):
            sb = exps[inds + i]
            dist, value, _ = self.acmodel(sb.env_steps)

            entropy = dist.entropy().mean()

            policy_loss = -(dist.log_prob(sb.agent_steps.actions) * sb.advantages).mean()

            value_loss = (value - sb.returnn).pow(2).mean()

            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

            # Update batch values

            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_loss += loss

        # Update update values

        update_entropy /= self.num_recurr_steps
        update_value /= self.num_recurr_steps
        update_policy_loss /= self.num_recurr_steps
        update_value_loss /= self.num_recurr_steps
        update_loss /= self.num_recurr_steps

        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs["entropy"] = update_entropy
        logs["value"] = update_value
        logs["policy_loss"] = update_policy_loss
        logs["value_loss"] = update_value_loss
        logs["grad_norm"] = update_grad_norm

        return logs

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


    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.num_recurr_steps)
        return starting_indexes
