#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import gym
import time

from torch_rl.envs_agents.snake import SnakeEnv

def run_manual(env):

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward

        elif keyName == 'SPACE':
            action = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.actions.drop

        elif keyName == 'RETURN':
            action = env.actions.done

        else:
            print("unknown key %s" % keyName)
            return

        obs, reward, done, info = env.step(action)

        print('step=%s, reward=%.2f' % (env.step_count, reward))

        if done:
            print('done!')
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    run_manual(SnakeEnv())
