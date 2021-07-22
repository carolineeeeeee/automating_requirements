import os
import sys
import json
import scipy
import torch
import logging
import argparse
import pathlib2
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from typing import List, Set, Dict, Tuple
from collections import defaultdict
from torch.utils.data import DataLoader
from robustbench.utils import load_model

from src.constant import *
from src.bootstrap import gen_bootstrap, gen_cifar10c_bootstrap
from src.evaluate import run_model, estimate_conf_int
from src.helper import read_cifar10_c_ground_truth


def run(num_batch: int, batch_size: int, data_path: str, gen_path: str, rq_type: str, model_name: str, corruption_type: str):
    ground_truth = read_cifar10_c_ground_truth(os.path.join(ROOT_PATH, data_path, corruption_type, "labels.txt"))
    df = gen_cifar10c_bootstrap(num_batch, gen_path, corruption_type, batch_size)
    df.to_csv(os.path.join(ROOT_PATH, "tmp1.csv"))
    record_df = run_model(model_name, df, cpu=False)
    conf, mu, sigma, satisfied = estimate_conf_int(record_df, rq_type, 1, ground_truth, 0.95)
    return conf, mu, sigma, satisfied


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Cifar10 Bootstrapping")
    parser.add_argument("--num_batch", type=int, default=10, help="Number of bootstrap iterations")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of unique images per bootstrap iteration")
    parser.add_argument("--corruption_type", type=str, required=True,
                        choices=CIFAR10_C_CORRUPTION, help="Corruption Type of Cifar10-C")
    parser.add_argument("--data_path", type=str, default=os.path.join(ROOT_PATH, "cifar10_c_data", "val"),
                        help="Path to dataset, should be a folder containing images and a labels.txt")
    parser.add_argument("--gen_path", type=str, default=os.path.join(ROOT_PATH, "cifar10c_bootstrap_output"),
                        help="location to save intermediate bootstrapping images")
    parser.add_argument("--rq_type", required=True, choices=['abs', 'rel'], help="requirement type")
    parser.add_argument("--model_name",  required=True, choices=ROBUSTBENCH_CIFAR10_MODEL_NAMES, help="Model Name")

    args = parser.parse_args()
    run(args.num_batch, args.batch_size, args.data_path, args.gen_path,
        args.rq_type, args.model_name, args.corruption_type)
