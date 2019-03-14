import argparse
import os
import pprint
import sys
sys.path.append(os.getcwd())


from torch_rl.plot_curves import plot_curves
import torch
from resmonres.monitor_system_parameters import MonitorSysParams

from torch_rl.envs_agents.snake import build_SnakeEnv, SnakeDQNAgent
from torch_rl.visualize import visualize_it
from torch_rl.algos.dqn import DQNAlgo
from torch_rl.algos.train_methods import CsvLogger
from torch_rl.utils import get_logger, get_csv_writer, set_seeds

if __name__ == '__main__':
    storage_path = os.getcwd() + '/storage'
    num_batches=100000
    def update_default_params(exp_specific_name,p):

        params = {
            'model_name': 'snake-ddqn-1e-0p-%db' % (num_batches),
            'seed': 3,
            'num_batches': num_batches,
            'num_rollout_steps': 1,
            'memory_size': 10000,
            'initial_eps_value': 0.8,
            'final_eps_value': 0.01,
            'end_of_interpolation': num_batches,

        }
        params['model_name']+=exp_specific_name
        params.update(p)
        return params

    experiments = [
        ('1kmem',{'memory_size':1000}),
        ('10kmem',{'memory_size':10000}),
        ('0.1to0.1eps',{'initial_eps_value':0.1,'final_eps_value':0.1}),
        ('0.1to0.01eps',{'initial_eps_value':0.1,'final_eps_value':0.01}),
        ('0.8to0.01eps',{'initial_eps_value':0.8,'final_eps_value':0.01}),

    ]
    for exp_specific_name, exp_params in experiments:
        params = update_default_params(exp_specific_name,exp_params)
        args = argparse.Namespace(**params)
        model_dir = storage_path + '/' + args.model_name

        logger = get_logger(model_dir)
        csv_file, csv_writer = get_csv_writer(model_dir)
        set_seeds(args.seed)

        # envs = build_CartPoleEnv(num_envs=args.num_envs, use_multiprocessing=False)
        # q_model = CartPoleAgent(envs.observation_space, envs.action_space)
        # target_model = CartPoleAgent(envs.observation_space, envs.action_space)

        envs = build_SnakeEnv(num_envs=1, num_processes=0)
        agent = SnakeDQNAgent(envs.observation_space, envs.action_space)
        target_model = SnakeDQNAgent(envs.observation_space, envs.action_space)

        logger.info("Parameters:\n"+pprint.pformat(params))

        logger.info("Model successfully created\n")
        logger.info("{}\n".format(agent))

        if torch.cuda.is_available():
            agent.cuda()
        logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

        algo = DQNAlgo(envs, agent, target_model, num_rollout_steps=args.num_rollout_steps,
                       lr=0.0001,double_dpn=True,memory_size=args.memory_size,
                       target_model_update_interval=20)

        with MonitorSysParams(model_dir):
            algo.train_model(args.num_batches,CsvLogger(csv_file,csv_writer,logger),
                             initial_eps_value=args.initial_eps_value,
                             final_eps_value=args.final_eps_value,
                             end_of_interpolation=args.end_of_interpolation
                             )
    # env = build_CartPoleEnv(num_envs=1, use_multiprocessing=False)
    # visualize_it(env,model_dir)
    # visualize_it(build_SnakeEnv(num_envs=1, num_processes=0),agent)
    plot_curves(storage_path)