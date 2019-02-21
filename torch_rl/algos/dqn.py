import time

import gym
import torch
import torch.nn.functional as F
import numpy

from agent_models import QModel
from torch_rl.algos.train_methods import flatten_parallel_rollout, log_step, ExperienceMemory, get_metrics_to_log, \
    CsvLogger
from torch_rl.utils.dictlist import DictList
from utils.general import calc_stats


def gather_exp_via_rollout(model_step_fun, env_step_fun, exp_memory:ExperienceMemory, num_rollout_steps):
    for _ in range(num_rollout_steps):
        i = exp_memory.last_written_idx
        env_step = env_step_fun(exp_memory[i].agent)
        agent_step = model_step_fun(env_step)
        exp_memory.store_single(DictList.build({'env':env_step,'agent':agent_step}))


class LinearAndConstantSchedule(object):

    def __init__(self, initial_value, final_value, end_of_interpolation):
        self.initial_value = initial_value
        self.final_value = final_value
        self.end_of_interpolation = end_of_interpolation

    def value(self, progress_indicator):
        def interpolate_linear_single(start, end, coefficient):
            return start + (end - start) * coefficient

        if progress_indicator <= self.end_of_interpolation:
            return interpolate_linear_single(
                self.initial_value, self.final_value, progress_indicator/self.end_of_interpolation
            )
        else:
            return self.final_value



class DQNAlgo(object):

    def __init__(self, env:gym.Env, model:QModel,target_model, num_rollout_steps=None, target_model_update_interval=10,
                 double_dpn=False,
                 discount=0.99, lr=7e-4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-5):

        self.double_dqn = double_dpn
        self.env = env
        self.model = model
        self.target_model = target_model
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()
        self.target_model_update_interval = target_model_update_interval


        self.num_rollout_steps = num_rollout_steps
        self.discount = discount
        self.lr = lr
        self.max_grad_norm=1.0
        self.batch_idx = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        initial_env_step = self.env.reset()
        self.num_envs = len(initial_env_step['reward'])
        with torch.no_grad():
            initial_agent_step = self.model.step(initial_env_step)
        initial_exp = DictList.build({'env':initial_env_step,'agent':initial_agent_step})

        self.exp_memory = ExperienceMemory(100, initial_exp,log_step)

        self.batch_size = self.num_rollout_steps * self.num_envs

        self.logged = {
            'rewards_sum':torch.zeros(self.num_envs, device=self.device),
            'num_steps_sum':torch.zeros(self.num_envs, device=self.device),
            'log_done_counter':0,
            'log_episode_rewards':[],
            'log_num_steps':[]
        }

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)

        with torch.no_grad():
            model.eval()
            def model_step_fun(env_step):
                return self.model.step(env_step,1.0)
            gather_exp_via_rollout(model_step_fun,self.env.step,self.exp_memory,len(self.exp_memory))


    def train_batch(self):
        with torch.no_grad():
            exps = self.collect_experiences()

            # if self.double_dqn:
            #     target_q = target_evaluator.get('model:q_next')
            #     model_q = evaluator.get('model:q_next')
            #     # Select largest 'target' value based on action that 'model' selects
            #     values = target_q.gather(1, model_q.argmax(dim=1, keepdim=True)).squeeze(1)
            # else:
            i = numpy.arange(0, len(exps)-1)
            next_i = numpy.arange(1, len(exps))

            next_env_steps = exps[next_i].env
            values = self.target_model(next_env_steps).max(dim=1)[0] # [0] is because in pytorch .max(...) returns tuple (max values, argmax)
            mask = torch.tensor((1 - exps[next_i].env.done), dtype=torch.float)
            estimated_return = exps[next_i].env.reward + self.discount * values * mask

        q_values = self.model(exps[i].env)
        exp_actions = exps[i].agent.actions.unsqueeze(1)
        q_selected = q_values.gather(1, exp_actions).squeeze(1)

        original_losses = F.smooth_l1_loss(q_selected, estimated_return, reduction='none')
        loss_value = torch.mean(original_losses)
        self.optimizer.zero_grad()
        loss_value.backward()

        # update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.model.parameters()) ** 0.5
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()


        self.batch_idx+=1
        if self.batch_idx % self.target_model_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

        return get_metrics_to_log(self.exp_memory.log,20)

    def collect_experiences(self):
        with torch.no_grad():
            self.model.eval()

            def model_step_fun(env_step):
                return self.model.step(env_step,self.eps_schedule.value(self.batch_idx))

            gather_exp_via_rollout(model_step_fun,self.env.step,self.exp_memory,self.num_rollout_steps)

        batch_exp = self.exp_memory.sample_batch(self.batch_size//self.num_envs)
        batch_exp = DictList.build(flatten_parallel_rollout(batch_exp))
        return batch_exp

    def train_model(self,num_batches,logger:CsvLogger):

        self.eps_schedule = LinearAndConstantSchedule(initial_value=1.0, final_value=0.1, end_of_interpolation=num_batches)
        logger.on_train_start()
        for k in range(num_batches):
            update_start_time = time.time()
            metrics_to_log = self.train_batch()
            update_end_time = time.time()
            logger.log_it(metrics_to_log,update_start_time,update_end_time)





