import torch
from multiprocessing import Process, Pipe
from typing import List, Callable

import gym

def worker(conn, env_supplier):
    env = env_supplier()
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        elif cmd == 'get_spaces':
            conn.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self,env_suppliers:List[Callable[[],gym.Env]] ):

        self.locals = []
        self.num_envs = len(env_suppliers)
        for env_sup in env_suppliers:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env_sup)) #TODO: pickling env to get it into worker-process is not acceptable!
            p.daemon = True
            p.start()
            remote.close()

        self.locals[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.locals[0].recv()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        return [local.recv() for local in self.locals]

    def step(self, actions):
        for local, action in zip(self.locals, actions):
            local.send(("step", action))
        obs, rewards,dones,infos = zip(*[local.recv() for local in self.locals])
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones, dtype=torch.float)
        return obs,rewards,dones, infos

    def render(self):
        raise NotImplementedError

    @staticmethod
    def build(env_name,num_envs,seed):
        def build_env_supplier(i):
            def env_supplier():
                env = gym.make(env_name)
                env.seed(seed + 10000 * i)
                return env

            return env_supplier

        return ParallelEnv([build_env_supplier(i) for i in range(num_envs)])
