import time
import utils

import torch

from torch_rl.algos.base import BaseAlgo


def train_model(num_steps_to_train, algo:BaseAlgo, logger, csv_writer, csv_file):

    num_steps = 0
    total_start_time = time.time()
    update = 0

    while num_steps < num_steps_to_train:

        update_start_time = time.time()
        logs = algo.update_parameters()
        update_end_time = time.time()

        num_steps += logs["num_frames"]
        update += 1


        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = int(time.time() - total_start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

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
