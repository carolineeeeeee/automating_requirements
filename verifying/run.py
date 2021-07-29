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
import matlab.engine

from src.constant import *
from src.bootstrap import gen_bootstrap
from src.evaluate import run_model, estimate_conf_int, obtain_preserved_min_degradation
from src.helper import read_cifar10_ground_truth, get_transformation_threshold

def preparation_and_bootstrap(num_batch: int, batch_size: int, data_path: str, gen_path: str, rq_type: str, transformation: str, threshold: float = None):
    ground_truth = read_cifar10_ground_truth(os.path.join(ROOT_PATH, data_path, "labels.txt"))
    if threshold is None:
        threshold = get_transformation_threshold(transformation, rq_type)
    df = gen_bootstrap(num_batch=num_batch, orig_path=data_path, gen_path=gen_path, t=threshold,
                       batch_size=batch_size, transformation=transformation)
    df.to_csv(os.path.join(ROOT_PATH, "tmp.csv"))
    return ground_truth, df

def run(ground_truth, boot_df, rq_type: str, model_name: str):
    record_df = run_model(model_name, boot_df)
    if rq_type == 'rel':
        a = obtain_preserved_min_degradation(record_df)
        conf, mu, sigma, satisfied = estimate_conf_int(record_df, rq_type, 1, ground_truth, a)
    elif rq_type == 'abs':
        conf, mu, sigma, satisfied = estimate_conf_int(record_df, rq_type, 1, ground_truth, 0.95)
    else:
        raise ValueError("Invalid rq_type")
    return conf, mu, sigma, satisfied


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Cifar10 Bootstrapping")
    parser.add_argument("--num_batch", type=int, default=10, help="Number of bootstrap iterations")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of unique images per bootstrap iteration")
    parser.add_argument("--transformation", type=str, required=True,
                        choices=TRANSFORMATIONS, help="Transformation to use")
    parser.add_argument("--data_path", type=str, default=os.path.join(ROOT_PATH, "cifar10_data", "val"),
                        help="Path to dataset, should be a folder containing images and a labels.txt")
    parser.add_argument("--gen_path", type=str, default=os.path.join(ROOT_PATH, "bootstrap_output"),
                        help="location to save intermediate bootstrapping images")
    parser.add_argument("--rq_type", required=True, choices=['abs', 'rel'], help="requirement type")
    parser.add_argument("--model_name",  required=True, choices=ROBUSTBENCH_CIFAR10_MODEL_NAMES, help="Model Name")

    args = parser.parse_args()
    run(args.num_batch, args.batch_size, args.transformation, args.data_path, args.gen_path, args.rq_type, args.model_name)
