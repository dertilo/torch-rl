#!/usr/bin/env python3

import argparse
import time
import torch

from torch_rl import utils
from torch_rl.penv import ParallelEnv

try:
    import gym_minigrid
except ImportError:
    pass
#TODO: !!!
assert False

def run_evaluation(model_file):
    # Set seed for all randomness sources
    utils.seed(args.seed)

    env = ParallelEnv.build(args.env,args.procs,args.seed)
    # Define agent
    model_dir = utils.get_model_dir(model_file)
    agent = utils.Agent(args.env, env.observation_space, model_dir, args.argmax, args.procs)
    print("CUDA available: {}\n".format(torch.cuda.is_available()))
    # Initialize logs
    logs = {"num_frames_per_episode": [], "return_per_episode": []}
    # Run the agent
    start_time = time.time()
    obss = env.reset()
    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=agent.device)
    log_episode_num_frames = torch.zeros(args.procs, device=agent.device)
    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss)
        obss, rewards, dones, _ = env.step(actions)
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=agent.device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=agent.device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=agent.device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask
    end_time = time.time()
    # Print logs
    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames / (end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
    print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
          .format(num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values()))
    # Print worst episodes
    n = args.worst_episodes_to_show
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        for i in indexes[:n]:
            print(
                "- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", default='MiniGrid-Empty-8x8-v0',
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default='MiniGrid-Empty-8x8-v0_a2c_seed1_19-02-09-09-59-41',
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="The number of worse episodes to show")
args = parser.parse_args()



run_evaluation('MiniGrid-Empty-8x8-v0_a2c_seed1_19-02-12-11-46-34')