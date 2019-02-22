import argparse
import time

import gym
import torch

from envs import build_SnakeEnv, SnakeAgent
from scripts.visualize import visualize_it
from torch_rl.algos.a2c import A2CAlgo
from utils.general import set_seeds, calc_stats
from utils.save import get_logger, get_csv_writer, save_model

try:
    import gym_minigrid
except ImportError:
    raise Exception('gym_minigrid must be in PYTHONPATH!')


def train_model(num_batches, algo, logger, csv_writer, csv_file):

    num_steps = 0
    total_start_time = time.time()
    update = 0

    for k in range(num_batches):

        update_start_time = time.time()
        logs = algo.update_parameters()
        update_end_time = time.time()

        num_steps += logs["num_frames"]
        update += 1


        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = int(time.time() - total_start_time)
        return_per_episode = calc_stats(logs["return_per_episode"])
        num_frames_per_episode = calc_stats(logs["num_frames_per_episode"])

        log_stuff(csv_file, csv_writer, duration, fps, logger, logs, num_frames_per_episode, num_steps,
                  return_per_episode, update)


def log_stuff(csv_file, csv_writer, duration, fps, logger, logs, num_frames_per_episode, num_steps, return_per_episode,
              update):
    header = ["update", "frames", "FPS", "duration"]
    data = [update, num_steps, fps, duration]
    header += ["rreturn_" + key for key in return_per_episode.keys()]
    data += return_per_episode.values()
    header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
    data += num_frames_per_episode.values()
    header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
    data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
    logger.info(
        "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
            .format(*data))
    header += ["return_" + key for key in return_per_episode.keys()]
    data += return_per_episode.values()
    if num_steps == 0:
        csv_writer.writerow(header)
    csv_writer.writerow(data)
    csv_file.flush()


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--algo", default='a2c',
                    help="algorithm to use: a2c | ppo (REQUIRED)")
env_name = 'MiniGrid-Snake-v0'
# env_name = 'MiniGrid-Empty-8x8-v0'
model_name = 'mlp-128-64'
args = argparse.Namespace(**{
    'model_name':model_name,
    'env_name':env_name,
    'seed':1,
    'num_envs':16,
    'num_batches':600,
    'num_rollout_steps':5

})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = 'storage/'+args.model_name

logger = get_logger(model_dir)
csv_file, csv_writer = get_csv_writer(model_dir)

set_seeds(args.seed)

envs = build_SnakeEnv(num_envs=16, use_multiprocessing=True)
agent = SnakeAgent(envs.observation_space, envs.action_space)
logger.info("Model successfully created\n")
logger.info("{}\n".format(agent))

if torch.cuda.is_available():
    agent.cuda()
logger.info("CUDA available: {}\n".format(torch.cuda.is_available()))

algo = A2CAlgo(envs, agent, num_rollout_steps=args.num_rollout_steps, discount=0.99, lr=7e-4, gae_lambda=0.95,
               entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, num_recurr_steps=1)

train_model(args.num_batches,algo,logger,csv_writer,csv_file)

# if torch.cuda.is_available():
#     agent.cpu()
# save_model(agent, model_dir)
# if torch.cuda.is_available():
#     agent.cuda()

visualize_it(build_SnakeEnv(num_envs=1, use_multiprocessing=False),agent)