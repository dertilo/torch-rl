import torch
from multiprocessing import Process, Pipe
from typing import List, Callable, Dict

import gym

def worker(conn, env_supplier):
    env = env_supplier()
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            conn.send(env.step(data))
        elif cmd == "reset":
            conn.send(env.reset())
        elif cmd == 'get_spaces':
            conn.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):

    def __init__(self,env_suppliers:List[Callable[[],gym.Env]] ):

        self.locals = []
        self.num_envs = len(env_suppliers)
        for env_sup in env_suppliers:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env_sup))
            p.daemon = True
            p.start()
            remote.close()

        self.locals[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.locals[0].recv()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        LD = [local.recv() for local in self.locals]
        DL = {k: [dic[k] for dic in LD] for k in LD[0]}
        return DL

    def step(self, actions):
        for local, action in zip(self.locals, actions):
            local.send(("step", action))

        LD = [local.recv() for local in self.locals]
        DL = {k: [dic[k] for dic in LD] for k in LD[0]}
        return DL

    def render(self):
        raise NotImplementedError

    @staticmethod
    def build(build_env_supplier, num_envs):
        return ParallelEnv([build_env_supplier(i) for i in range(num_envs)])

class SingleEnvWrapper(gym.Env):
    def __init__(self,env:gym.Env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        return {k:[v] for k,v in self.env.step(action).items()}

    def reset(self):
        return {k:[v] for k,v in self.env.reset().items()}

    def render(self, mode='human'):
        return self.env.render(mode)