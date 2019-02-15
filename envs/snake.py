import random
from collections import deque
from enum import IntEnum
from typing import List, Tuple

import numpy
from gym import spaces
from gym.envs import register
from gym_minigrid.minigrid import MiniGridEnv, Goal, Lava, Grid


class Snake(object):
    def __init__(self,parts:List[Tuple[int,int]]):
        self.body = deque(parts)

    def rm_tail(self):
        return self.body.pop()

    def grow_head(self, x, y):
        self.body.appendleft((x,y))


class SnakeEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(self, size=9):

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )
        self.actions = SnakeEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))



    def spawn_new_food(self):
        empties = [(i,j) for i in range(self.grid.height) for j in range(self.grid.width) if self.grid.get(i,j) is None]
        self.grid.set(*random.choice(empties),Goal())

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.start_pos = (2, 2)
        self.start_dir = 0
        self.snake = Snake([self.start_pos,(self.start_pos[0],self.start_pos[1]-1)])


        # Place a goal square in the bottom-right corner
        self.grid.set(width - 2, height - 2, Goal())

        self.mission = "get to the green goal square"

    def reset(self):
        return self.perceive(super().reset())

    def step(self, action):
        self.step_count += 1

        done = False

        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        elif action == self.actions.forward:
            pass
        else:
            assert False, "unknown action: %d"%action

        fwd_pos = self.agent_pos + self.dir_vec
        fwd_cell = self.grid.get(*fwd_pos)

        if fwd_cell is None:
            self.grid.set(*self.agent_pos, Lava())
            self.snake.grow_head(*self.agent_pos)
            self.grid.set(*self.snake.rm_tail(), None)
            self.agent_pos = fwd_pos

            reward = -0.001

        elif fwd_cell.type == 'goal':
            self.grid.set(*self.agent_pos, Lava())
            self.snake.grow_head(*self.agent_pos)
            self.agent_pos = fwd_pos

            self.spawn_new_food()
            reward = 1.0

        elif (fwd_cell.type == 'lava' or fwd_cell.type == 'wall'):
            reward = .0
            done = True

        else:
            assert False

        if self.step_count >= self.max_steps:
            done = True

        if self.step_count==1 and done:
            assert False

        obs = self.gen_obs()

        obs = self.perceive(obs)

        return obs, reward, done, {}

    def perceive(self, obs):
        return obs


register(
    id='MiniGrid-Snake-v0',
    entry_point='envs:SnakeEnv'
)


