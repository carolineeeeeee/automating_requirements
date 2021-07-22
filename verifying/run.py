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
from src.bootstrap import gen_bootstrap
from src.evaluate import run_model, estimate_conf_int
from src.helper import read_cifar10_ground_truth

if __name__ == "__main__":

    ground_truth = read_cifar10_ground_truth("./cifar10_data/val/labels.txt")

    df = gen_bootstrap(num_batch=2, orig_path="./cifar10_data/val", gen_path="./bootstrap_output", t=0.85,
                       save_name="_val", batch_size=10, transformation=UNIFORM_NOISE)
    # print(df)
    # print(df.columns)
    record_df = run_model('Standard', df)
    # print(record_df)
    # print(record_df.columns)
    # estimate_conf_int(record_df, 'abs', 1, ground_truth)
    estimate_conf_int(record_df, 'abs', 1, ground_truth, 0.95)
