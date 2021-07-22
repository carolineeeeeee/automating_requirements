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
__root__ = pathlib2.Path(__file__).absolute().parent


if __name__ == '__main__':
    df = gen_bootstrap(num_batch=2, orig_path="./cifar10_data/val", gen_path="./bootstrap_output", t=0.85,
                       save_name="_val", batch_size=10, transformation=MOTION_BLUR)
    error = []
    for tran in TRANSFORMATIONS:
        print(tran)
        try:
            df = gen_bootstrap(num_batch=2, orig_path="./cifar10_data/val", gen_path="./bootstrap_output", t=0.85,
                               save_name="_val", batch_size=10, transformation=tran)
        except Exception as e:
            print(e)
            error.append(tran)
    print(error)
