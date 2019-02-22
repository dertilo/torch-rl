import time
from collections import Counter

import gym
import torch
import torch.nn.functional as F
import numpy

from agent_models import QModel
from torch_rl.algos.train_methods import flatten_parallel_rollout, log_step, ExperienceMemory, \
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
        self.max_grad_norm=10.0
        self.batch_idx = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        initial_env_step = self.env.reset()
        self.num_envs = len(initial_env_step['reward'])
        with torch.no_grad():
            initial_agent_step = self.model.step(initial_env_step)
        initial_exp = DictList.build({'env':initial_env_step,'agent':initial_agent_step})

        self.exp_memory = ExperienceMemory(200, initial_exp,log_step)

        self.batch_size = 32#self.num_rollout_steps * self.num_envs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

        with torch.no_grad():
            model.eval()
            def model_step_fun(env_step):
                return self.model.step(env_step,1.0)
            def env_step_fun(act):
                # env.render()
                return self.env.step(act)
            gather_exp_via_rollout(model_step_fun,env_step_fun,self.exp_memory,len(self.exp_memory))

        # done_obs = self.exp_memory.buffer[:-1].env.observation[self.exp_memory.buffer[1:].env.done, :]
        # x = done_obs[:,0]
        # theta = done_obs[:,2]
        # done =  x.lt(-self.env.env.env.env.env.x_threshold) \
        #         + x.gt(self.env.env.env.env.env.x_threshold) \
        #         + theta.lt(-self.env.env.env.env.env.theta_threshold_radians) \
        #         + theta.gt(self.env.env.env.env.env.theta_threshold_radians)
        # print(Counter(self.exp_memory.buffer.agent.actions.squeeze().numpy().tolist()))

    def train_batch(self):
        from matplotlib import pyplot as plt
        with torch.no_grad():
            self.model.eval()
            exps = self.collect_experiences()
            print(Counter(self.exp_memory.buffer.agent.actions.squeeze().numpy().tolist()))
            # if self.double_dqn:
            #     target_q = target_evaluator.get('model:q_next')
            #     model_q = evaluator.get('model:q_next')
            #     # Select largest 'target' value based on action that 'model' selects
            #     values = target_q.gather(1, model_q.argmax(dim=1, keepdim=True)).squeeze(1)
            # else:
            done = exps.next_env_step.done
            # x = exps.env_step[done].observation[:,0]
            # theta = exps.env_step[done].observation[:,2]
            # plt.plot(x.numpy(),theta.numpy(),'.');
            print('%d of %d are done'%(torch.sum(done).numpy(),done.shape[0]))

            next_env_steps = exps.next_env_step
            max_next_value = self.model(next_env_steps).max(dim=1)[0] # [0] is because in pytorch .max(...) returns tuple (max values, argmax)
            max_next_value = torch.tensor(max_next_value.data)
            mask = torch.tensor((1 - next_env_steps.done), dtype=torch.float)
            estimated_return = next_env_steps.reward + self.discount * max_next_value * mask

        self.model.train()
        q_values = self.model(exps.env_step)
        exp_actions = exps.action.unsqueeze(1)
        q_selected = q_values.gather(1, exp_actions).squeeze(1)

        original_losses = F.mse_loss(q_selected, estimated_return, reduction='none')
        # plt.plot(self.batch_idx*numpy.ones((50,)),original_losses.data.numpy(),'.')
        loss_value = torch.mean(original_losses)
        self.optimizer.zero_grad()
        loss_value.backward()

        # update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.model.parameters()) ** 0.5
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()


        self.batch_idx+=1
        if self.batch_idx % self.target_model_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

        def get_metrics_to_log(log,last_n_steps):
            keep = min(len(log['log_episode_rewards']), last_n_steps)
            metrics = {
                "dones":log['log_done_counter'],
                "loss":loss_value.data.numpy(),
                "rewards": calc_stats(log['log_episode_rewards'][-keep:])['mean'],
                "step": calc_stats(log['log_num_steps'][-keep:])['mean'],
            }
            return metrics

        return get_metrics_to_log(self.exp_memory.log,20)

    def collect_experiences(self):

        def model_step_fun(env_step):
            eps = self.eps_schedule.value(self.batch_idx)
            return self.model.step(env_step, eps)

        gather_exp_via_rollout(model_step_fun,self.env.step,self.exp_memory,self.num_rollout_steps)

        indexes = torch.randint(0,len(self.exp_memory)-1,(self.batch_size//self.num_envs,))
        next_indexes = indexes+1
        batch_exp = DictList.build(flatten_parallel_rollout({'env_step':self.exp_memory.buffer[indexes].env,
                                                             'action':self.exp_memory.buffer[indexes].agent.actions,
                                                             'next_env_step':self.exp_memory.buffer[next_indexes].env}))
        return batch_exp

    def train_model(self,num_batches,logger:CsvLogger):

        self.eps_schedule = LinearAndConstantSchedule(initial_value=0.1, final_value=0.1, end_of_interpolation=num_batches)
        logger.on_train_start()
        for k in range(num_batches):
            update_start_time = time.time()
            metrics_to_log = self.train_batch()
            update_end_time = time.time()
            logger.log_it(metrics_to_log,update_start_time,update_end_time)





