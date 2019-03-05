import argparse
import os

import torch
from resmonres.monitor_system_parameters import MonitorSysParams

from torch_rl.envs_agents.snake import build_SnakeEnv, SnakeDQNAgent
from torch_rl.visualize import visualize_it
from torch_rl.algos.dqn import DQNAlgo
from torch_rl.algos.train_methods import CsvLogger
from torch_rl.utils import get_logger, get_csv_writer, set_seeds

def main():

    model_name = 'snake-ddqn-1e-0p'
    args = argparse.Namespace(**{
        'model_name':model_name,
        'seed':1,
        'num_batches':6000,
        'num_rollout_steps':1

    })
    model_dir = 'storage/'+args.model_name

    logger = get_logger(model_dir)
    csv_file, csv_writer = get_csv_writer(model_dir)
    set_seeds(args.seed)

    # envs = build_CartPoleEnv(num_envs=args.num_envs, use_multiprocessing=False)
    # q_model = CartPoleAgent(envs.observation_space, envs.action_space)
    # target_model = CartPoleAgent(envs.observation_space, envs.action_space)

    envs = build_SnakeEnv(num_envs=1, num_processes=0)
    agent = SnakeDQNAgent(envs.observation_space, envs.action_space)
    target_model = SnakeDQNAgent(envs.observation_space, envs.action_space)

    logger.info("Model successfully created\n")
    logger.info("{}\n".format(agent))

    if torch.cuda.is_available():
        agent.cuda()
    logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

    algo = DQNAlgo(envs, agent, target_model, num_rollout_steps=args.num_rollout_steps,
                   lr=0.0001,double_dpn=True,
                   target_model_update_interval=20)
    cwd = os.getcwd()
    with MonitorSysParams(cwd):
        algo.train_model(args.num_batches,CsvLogger(csv_file,csv_writer,logger))
    # env = build_CartPoleEnv(num_envs=1, use_multiprocessing=False)
    # visualize_it(env,model_dir)
    # visualize_it(build_SnakeEnv(num_envs=1, num_processes=0),agent)

if __name__ == '__main__':
    main()