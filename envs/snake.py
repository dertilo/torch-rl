import random
from collections import deque
from enum import IntEnum
from typing import List, Tuple, Dict, Any

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from gym_minigrid.minigrid import MiniGridEnv, Goal, Lava, Grid
from torch.distributions.categorical import Categorical

from agent_models import initialize_parameters, ACModel
from scripts.visualize import visualize_it
from torch_rl.utils.penv import SingleEnvWrapper, ParallelEnv
from utils.format import preprocess_images


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
        return super().reset()

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

        return obs, reward, done, {}

class SnakeAgent(ACModel):
    def __init__(self, obs_space, action_space, use_memory=False):
        super().__init__()
        self.hidden_state = None
        self.has_hiddenstate = use_memory

        # Define image embedding
        # self.visual_nn = nn.Sequential(
        #     nn.Conv2d(3, 16, (2, 2)),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #     nn.Conv2d(16, 32, (2, 2)),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, (2, 2)),
        #     nn.ReLU()
        # )
        # n = obs_space["image"][0]
        # m = obs_space["image"][1]
        # self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        self.image_embedding_size = 64#((n-1)//2-2)*((m-1)//2-2)*64
        image_shape = obs_space.spaces['image'].shape
        self.visual_nn = nn.Sequential(
            *[
                nn.Linear(image_shape[0]*image_shape[1], 128),
                nn.ReLU(),
                nn.Linear(128, self.image_embedding_size),
                nn.ReLU(),
             ]
        )

        # Define memory
        if self.has_hiddenstate:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        # if self.use_text:
        #     self.embedding_size += self.text_embedding_size

        # Define actor's model
        if isinstance(action_space, gym.spaces.Discrete):
            self.actor = nn.Sequential(
                # nn.Linear(self.embedding_size, 64),
                # nn.Tanh(),
                nn.Linear(self.image_embedding_size, action_space.n)
            )
        else:
            raise ValueError("Unknown action space: " + str(action_space))

        # Define critic's model
        self.critic = nn.Sequential(
            # nn.Linear(self.embedding_size, 64),
            # nn.Tanh(),
            nn.Linear(self.image_embedding_size, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    @property
    def hiddenstate_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, observation, memory=None):
        if 'done' in observation:
            self.reset_hidden_state((1-observation.get('done')).unsqueeze(1))

        image = observation.get('image')
        if memory is None:
            memory = self.hidden_state
        x = image[:,:,:,0].view(image.size(0),-1)
        # x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        x = self.visual_nn(x)
        x = x.reshape(x.shape[0], -1)

        if self.has_hiddenstate:
            assert False
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def set_hidden_state(self,agent_step):
        self.hidden_state = agent_step.get('hidden_states')

    def reset_hidden_state(self,mask):
        if self.hidden_state is None or self.hidden_state.shape != mask.shape:
            self.hidden_state = torch.zeros(mask.shape[0], self.hiddenstate_size)
        else:
            self.hidden_state = self.hidden_state * mask

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

class PreprocessWrapper(gym.Env):
    def __init__(self,env:gym.Env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, act):
        actions = act.get('actions').cpu().numpy()
        env_step = self.env.step(actions)
        return self.process_env_step(env_step)

    def process_env_step(self, env_step):
        return {'reward': torch.tensor(env_step.get('reward'),dtype=torch.float),
                'done': torch.tensor(env_step.get('done')),
                'image': preprocess_images(env_step.get('observation'))}

    def reset(self):
        env_step = self.env.reset()
        return self.process_env_step(env_step)

    def render(self, mode='human'):
        return self.env.render(mode)

class SnakeWrapper(gym.Env):
    def __init__(self):
        self.env = SnakeEnv()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action):
        obs,reward,done,_ = self.env.step(action)
        if done:
            obs = self.env.reset()
        return {'observation':obs['image'],'reward':reward,'done':done}

    def reset(self):
        obs = self.env.reset()
        return {'observation':obs['image'],'reward':0,'done':False}

    def render(self, mode='human'):
        return self.env.render(mode)

    def seed(self, seed=None):
        return self.env.seed(seed)


#
# register(
#     id='MiniGrid-Snake-v0',
#     entry_point='envs:SnakeEnv'
# )


def build_SnakeEnv(num_envs,use_multiprocessing):
    if not use_multiprocessing:
        assert num_envs==1
        env = PreprocessWrapper(SingleEnvWrapper(SnakeWrapper()))
    else:

        def build_env_supplier(i):
            def env_supplier():
                env = SnakeWrapper()
                env.seed(1000+i)
                return env

            return env_supplier

        env = PreprocessWrapper(ParallelEnv.build(build_env_supplier,num_envs))
    return env


if __name__ == '__main__':
    env = build_SnakeEnv(num_envs=1,use_multiprocessing=False)
    x = env.reset()
    agent = SnakeAgent(env.observation_space,env.action_space)
    visualize_it(env,agent)


