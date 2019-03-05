import gym
import numpy
import torch

from torch_rl.algos.abstract_agents import ACModel
from torch_rl.algos.train_methods import flatten_parallel_rollout, flatten_array, gather_exp_via_rollout, \
    ExperienceMemory, step_logging_fun, CsvLogger
from torch_rl.dictlist import DictList
from torch_rl.utils import calc_stats


def generalized_advantage_estimation(rewards,values,dones,num_rollout_steps,discount,gae_lambda):
    assert values.shape[0] == 1 + num_rollout_steps
    advantage_buffer = torch.zeros(rewards.shape[0]-1,rewards.shape[1])
    next_advantage = 0
    for i in reversed(range(num_rollout_steps)):
        mask = torch.tensor((1 - dones[i + 1]),dtype=torch.float32)
        bellman_delta = rewards[i+1] + discount * values[i + 1] * mask - values[i]
        advantage_buffer[i] = bellman_delta + discount * gae_lambda * next_advantage * mask
        next_advantage = advantage_buffer[i]
    return advantage_buffer

class A2CAlgo(object):

    def __init__(self, env:gym.Env, agent:ACModel, num_rollout_steps=None, discount=0.99, lr=7e-4, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, num_recurr_steps=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-5):

        self.env = env
        self.agent = agent
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

        with torch.no_grad():
            initial_agent_step = self.agent.step(initial_env_step)

        initial_exp = DictList.build({'env':initial_env_step,'agent':initial_agent_step})
        self.exp_memory = ExperienceMemory(self.num_rollout_steps+1, initial_exp,step_logging_fun)
        gather_exp_via_rollout(self.env.step,self.agent.step,self.exp_memory,self.num_rollout_steps)

        self.num_frames = self.num_rollout_steps * self.num_envs

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr)#
                                             # alpha=rmsprop_alpha, eps=rmsprop_eps)

    def train_batch(self):
        with torch.no_grad():
            exps = self.collect_experiences()

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        self.agent.train()

        inds = numpy.arange(0, self.num_frames, self.num_recurr_steps)
        self.agent.set_hidden_state(exps[inds].agent_steps)
        for i in range(self.num_recurr_steps):
            sb = exps[inds + i]
            dist, value, _ = self.agent(sb.env_steps)

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
        # update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.agent.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()

        def get_metrics_to_log(log,last_n_steps):
            keep = min(len(log['log_episode_rewards']), last_n_steps)
            metrics = {
                "episodes":log['log_done_counter'],
                "rewards": calc_stats(log['log_episode_rewards'][-keep:])['mean'],
                "med-rewards": calc_stats(log['log_episode_rewards'][-keep:])['median'],
                "episode-length": calc_stats(log['log_num_steps'][-keep:])['median'],
            }
            return metrics

        return get_metrics_to_log(self.exp_memory.log,20)

    def collect_experiences(self):
        self.agent.set_hidden_state(self.exp_memory[-1])
        assert self.exp_memory.current_idx==0
        self.exp_memory.last_becomes_first()
        gather_exp_via_rollout(self.env.step, self.agent.step,self.exp_memory,self.num_rollout_steps)
        assert self.exp_memory.last_written_idx == self.num_rollout_steps

        env_steps = self.exp_memory.buffer.env
        agent_steps = self.exp_memory.buffer.agent
        advantages = generalized_advantage_estimation(
            rewards=env_steps.reward,
            values=agent_steps.v_values,
            dones=env_steps.done,
            num_rollout_steps=self.num_rollout_steps,discount=self.discount,gae_lambda=self.gae_lambda
        )
        return DictList(**{
            'env_steps':DictList(**flatten_parallel_rollout(env_steps[:-1])),
            'agent_steps':DictList(**flatten_parallel_rollout(agent_steps[:-1])),
            'advantages': flatten_array(advantages),
            'returnn': flatten_array(agent_steps[:-1].v_values + advantages)
        })

    def train_model(self,num_batches,logger:CsvLogger):
        logger.on_train_start()
        for k in range(num_batches):
            metrics_to_log = self.train_batch()
            logger.log_it(metrics_to_log)

