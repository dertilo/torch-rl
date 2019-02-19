import csv
import os
import torch
import json
import logging
import sys

import utils
from utils.general import create_folders_if_necessary


def get_model_path(model_dir):
    return os.path.join(model_dir, "model.pt")

def load_model(model_dir):
    path = get_model_path(model_dir)
    model = torch.load(path)
    model.eval()
    return model

def save_model(model, model_dir):
    path = get_model_path(model_dir)
    create_folders_if_necessary(path)
    torch.save(model, path)

def get_log_path(model_dir):
    return os.path.join(model_dir, "log.txt")

def get_logger(model_dir):
    path = get_log_path(model_dir)
    create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()

def get_vocab_path(model_dir):
    return os.path.join(model_dir, "vocab.json")

def get_csv_path(model_dir):
    return os.path.join(model_dir, "log.csv")

def get_csv_writer(model_dir):
    csv_path = get_csv_path(model_dir)
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)