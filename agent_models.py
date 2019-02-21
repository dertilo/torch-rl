from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gym

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py


def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ACModel(nn.Module):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()
        self.hidden_state = None
        # Decide which components are enabled
        self.use_text = use_text
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

    def step(self,observation:Dict[str,torch.Tensor],hidden_state=None,argmax=False)->Dict[str,Any]:
        dist, values, self.hidden_state = self(observation, hidden_state if hidden_state is not None else self.hidden_state)

        if argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        logprob = dist.log_prob(actions)

        return {'actions':actions,'values':values,'logprobs':logprob,'hidden_states':self.hidden_state}

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


def epsgreedy_action(num_actions, policy_actions, epsilon):
    random_actions = torch.randint_like(policy_actions, num_actions)
    selector = torch.rand_like(random_actions, dtype=torch.float32)
    return torch.where(selector > epsilon, policy_actions, random_actions)

class QModel(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()

        self.image_embedding_size = 64
        image_shape = obs_space.spaces['image'].shape
        self.visual_nn = nn.Sequential(
            *[
                nn.Linear(image_shape[0]*image_shape[1], 128),
                nn.ReLU(),
                nn.Linear(128, self.image_embedding_size),
                nn.ReLU(),
             ]
        )

        self.num_actions = action_space.n
        self.q_head = nn.Sequential(
            nn.Linear(self.image_embedding_size, self.num_actions)
        )

        self.apply(initialize_parameters)


    def forward(self, observation):

        image = observation.get('image')
        x = image[:,:,:,0].view(image.size(0),-1)
        x = self.visual_nn(x)
        x = x.reshape(x.shape[0], -1)

        q_values = self.q_head(x).squeeze(1)
        return q_values

    def step(self,observation:Dict[str,torch.Tensor],eps=1.0)->Dict[str,Any]:
        q_values = self(observation)
        policy_actions = q_values.argmax(dim=1)
        actions = epsgreedy_action(self.num_actions, policy_actions, eps)
        return {'actions':actions,'q_values':q_values}