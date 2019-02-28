#!/usr/bin/env python3

import argparse
import gym
import time

from torch_rl.utils import set_seeds

try:
    import gym_minigrid
except ImportError:
    pass

def visualize_it(env:gym.Env,agent,pause_dur=0.1,seed=0,argmax=False,num_steps=1000):

    set_seeds(seed)
    env.seed(seed)
    c=0
    obs = env.reset()
    while True:
        c+=1
        time.sleep(pause_dur)
        renderer = env.render()

        action = agent.step(obs)
        obs = env.step(action)

        if renderer.window is None:
            break
        if c>num_steps:
            break

if __name__ == '__main__':
    # Parse arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='MiniGrid-Snake-v0',
                        help="name of the environment to be run (REQUIRED)")
    parser.add_argument("--model", default='storage/mlp-128-64',
                        help="name of the trained model (REQUIRED)")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed (default: 0)")
    parser.add_argument("--argmax", action="store_true", default=False,
                        help="select the action with highest probability")
    parser.add_argument("--pause", type=float, default=0.1,
                        help="pause duration between two consequent actions of the agent")
    args = parser.parse_args()

    visualize_it(args.env,args.model,args.pause,args.seed,args.argmax)