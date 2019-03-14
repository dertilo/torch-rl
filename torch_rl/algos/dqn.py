import gym
import torch
import torch.nn.functional as F

from torch_rl.algos.train_methods import flatten_parallel_rollout, step_logging_fun, ExperienceMemory, \
    CsvLogger, gather_exp_via_rollout
from torch_rl.dictlist import DictList
from torch_rl.algos.abstract_agents import QModel
from torch_rl.utils import calc_stats


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

    def __init__(self, env:gym.Env, agent:QModel, target_model, num_rollout_steps=None, target_model_update_interval=10,
                 double_dpn=False,
                 discount=0.99, lr=7e-4,
                 rmsprop_alpha=0.99,
                 memory_size = 1000,
                 rmsprop_eps=1e-5):

        self.double_dqn = double_dpn
        self.env = env
        self.agent = agent
        self.target_model = target_model
        self.target_model.load_state_dict(agent.state_dict())
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
            initial_agent_step = self.agent.step(initial_env_step)
        initial_exp = DictList.build({'env':initial_env_step,'agent':initial_agent_step})

        self.batch_size = 32#self.num_rollout_steps * self.num_envs
        self.exp_memory = ExperienceMemory(memory_size//self.num_envs, initial_exp, step_logging_fun)


        self.optimizer = torch.optim.RMSprop(self.agent.parameters(), lr,alpha=rmsprop_alpha,eps=rmsprop_eps)

        with torch.no_grad():
            agent.eval()
            def agent_step_fun(env_step):
                return self.agent.step(env_step, 1.0)
            def env_step_fun(act):
                # env.render()
                return self.env.step(act)
            gather_exp_via_rollout(env_step_fun,agent_step_fun, self.exp_memory, len(self.exp_memory))

        # done_obs = self.exp_memory.buffer[:-1].env.observation[self.exp_memory.buffer[1:].env.done, :]
        # x = done_obs[:,0]
        # theta = done_obs[:,2]
        # done =  x.lt(-self.env.env.env.env.env.x_threshold) \
        #         + x.gt(self.env.env.env.env.env.x_threshold) \
        #         + theta.lt(-self.env.env.env.env.env.theta_threshold_radians) \
        #         + theta.gt(self.env.env.env.env.env.theta_threshold_radians)
        # print(Counter(self.exp_memory.buffer.agent.actions.squeeze().numpy().tolist()))

    def train_batch(self):
        with torch.no_grad():
            self.agent.eval()
            exps = self.collect_experiences()
            # print(Counter(self.exp_memory.buffer.agent.actions.squeeze().numpy().tolist()))
            next_env_steps = exps.next_env_step

            if self.double_dqn:
                target_q = self.target_model(next_env_steps)
                model_q = self.agent(next_env_steps)
                max_next_value = target_q.gather(1, model_q.argmax(dim=1, keepdim=True)).squeeze(1)
            else:
                max_next_value = self.target_model(next_env_steps).max(dim=1)[0] # [0] is because in pytorch .max(...) returns tuple (max values, argmax)

            mask = torch.tensor((1 - next_env_steps.done), dtype=torch.float)
            estimated_return = next_env_steps.reward + self.discount * max_next_value * mask

        self.agent.train()
        q_values = self.agent(exps.env_step)
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
            self.target_model.load_state_dict(self.agent.state_dict())
            self.target_model.eval()

        def get_metrics_to_log(log,last_n_steps):
            keep = min(len(log['log_episode_rewards']), last_n_steps)
            metrics = {
                "episodes":log['log_done_counter'],
                # "loss":loss_value.data.numpy(),
                "rewards": calc_stats(log['log_episode_rewards'][-keep:])['median'],
                "episode-length": calc_stats(log['log_num_steps'][-keep:])['median'],
                # "episode-length-std": calc_stats(log['log_num_steps'][-keep:])['std'],
            }
            return metrics

        return get_metrics_to_log(self.exp_memory.log,20)

    def collect_experiences(self):

        def agent_step_fun(env_step):
            eps = self.eps_schedule.value(self.batch_idx)
            return self.agent.step(env_step, eps)

        gather_exp_via_rollout(self.env.step,agent_step_fun, self.exp_memory, self.num_rollout_steps)

        indexes = torch.randint(0,len(self.exp_memory)-1,(self.batch_size//self.num_envs,))
        next_indexes = indexes+1
        batch_exp = DictList.build(flatten_parallel_rollout({'env_step':self.exp_memory.buffer[indexes].env,
                                                             'action':self.exp_memory.buffer[indexes].agent.actions,
                                                             'next_env_step':self.exp_memory.buffer[next_indexes].env}))
        return batch_exp

    def train_model(self,num_batches,logger:CsvLogger,initial_eps_value=1.0,final_eps_value=0.01,end_of_interpolation=None):

        e_o_i = num_batches if end_of_interpolation is None else end_of_interpolation
        self.eps_schedule = LinearAndConstantSchedule(initial_value=initial_eps_value,
                                                      final_value=final_eps_value,
                                                      end_of_interpolation=e_o_i)
        logger.on_train_start()
        for k in range(num_batches):
            metrics_to_log = self.train_batch()
            logger.log_it(metrics_to_log)





