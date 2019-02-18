#!/usr/bin/env python3

import argparse
import gym
import time

try:
    import gym_minigrid
except ImportError:
    pass

import utils
import envs


def visualize_it(env_name,model_file,pause_dur=0.1,seed=0,shift=0,argmax=False):
    # Set seed for all randomness sources
    utils.seed(seed)
    # Generate environment
    env = gym.make(env_name)
    env.seed(seed)
    for _ in range(shift):
        env.reset()
    # Define agent
    # model_dir = utils.get_model_dir(model_file)
    agent = utils.Agent(env_name, env.observation_space, model_file, argmax)
    # Run the agent
    done = True
    while True:
        if done:
            obs = env.reset()

        time.sleep(pause_dur)
        renderer = env.render()

        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)

        if renderer.window is None:
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
    parser.add_argument("--shift", type=int, default=0,
                        help="number of times the environment is reset at the beginning (default: 0)")
    parser.add_argument("--argmax", action="store_true", default=False,
                        help="select the action with highest probability")
    parser.add_argument("--pause", type=float, default=0.1,
                        help="pause duration between two consequent actions of the agent")
    args = parser.parse_args()

    visualize_it(args.env,args.model,args.pause,args.seed,args.shift,args.argmax)