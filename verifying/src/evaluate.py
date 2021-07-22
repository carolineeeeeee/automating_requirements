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

from src.dataset import Cifar10Dataset
from src.helper import get_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


def run_model(model_name: str, img_df: pd.DataFrame, cpu: bool = False):
    model = get_model(model_name, pretrained=True, val=True)
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    logger.info(f"Device: {str(device)}")
    model.to(device)
    batches = img_df['batch_id'].unique()
    pbar = tqdm(total=len(img_df))
    prediction_records = []
    err_top_5 = 0
    err_top_1 = 0
    total_image = 0
    for batch_id in batches:
        df = img_df[img_df['batch_id'] == batch_id]
        dataset = Cifar10Dataset(df)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=mp.cpu_count())

        for i, data in enumerate(dataloader):
            original_images = data['original_image'].to(device)
            transformed_images = data['transformed_image'].to(device)
            labels = data['label'].to(device)
            transformed_result = model(transformed_images)
            original_result = model(original_images)
            transformed_pred = torch.argmax(transformed_result, dim=1)
            original_pred = torch.argmax(original_result, dim=1)
            actual_batch_size = len(transformed_pred)
            top_1_error_count = actual_batch_size - int(torch.sum(labels == transformed_pred))
            top_5_error_count = actual_batch_size - int(
                torch.sum(torch.sum(torch.eq(labels.unsqueeze(1).repeat(1, 5), transformed_pred.topk(5).indices),
                                    dim=1)))
            total_image += actual_batch_size
            transformed_pred = transformed_pred.tolist()
            original_pred = original_pred.tolist()
            # print(data.keys())
            # print(data)
            for k, label in enumerate(labels.tolist()):
                prediction_records.append({
                    'model_name': model_name,
                    'bootstrap_itr_id': batch_id,
                    'dataload_itr_id': i,
                    'image_idx_in_batch': k,
                    'label': label,
                    'transformed_prediction': transformed_pred[k],
                    'original_prediction': original_pred[k],
                    'transformation': data['transformation'][k],
                    'original_filename': data['original_filename'][k],
                    'transformed_filename': data['filename'][k],
                    'original_path': data['original_path'][k],
                    'transformed_path': data['transformed_path'][k],
                })
            pbar.set_postfix({f'batch': batch_id})
            pbar.update(actual_batch_size)
    records_df = pd.DataFrame(data=prediction_records)
    records_df.to_csv("records.csv")
    return records_df


def calculate_confidence(acc_list, base_acc, req_acc):
    # fitting a normal distribution
    _, bins, _ = plt.hist(acc_list, 20, alpha=0.5, density=True)
    mu, sigma = scipy.stats.norm.fit(acc_list)
    best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
    print("Estimated mean from bootstrapping: " + str(mu))
    print("Estimated sigma from bootstrapping: " + str(sigma))
    # exit()
    distribution = scipy.stats.norm(mu, sigma)
    result = 1 - distribution.cdf(req_acc)
    print('confidence of satisfication:' + str(result))
    if result >= 0.5:
        print("requirement SATISFIED")
    else:
        print("requirement NOT SATISFIED")
    return result, mu, sigma, result >= 0.5


def estimate_conf_int(model_df: pd.DataFrame, rq_type: str, target_label_id: int, ground_truth: Dict, a: float = 0.95):
    """
    model_df should have the following columns:
    - model_name
    - bootstrap_itr_id
    - dataload_itr_id
    - image_idx_in_batch
    - label
    - transformed_prediction
    - original_prediction
    - transformation
    - original_filename
    - transformed_filename
    - original_path
    - transformed_path
    """
    # model_df
    model_df.drop_duplicates(subset=['original_filename']).set_index('original_filename')
    orig_results = {row['original_filename']: row['original_prediction'] for i, row in model_df.iterrows()}
    batch_ids = model_df['bootstrap_itr_id'].unique()
    batch_results = {batch_id: [(row['original_filename'], row['transformed_prediction'])
                                for i, row in model_df[model_df['bootstrap_itr_id'] == batch_id].iterrows()] for batch_id in batch_ids}
    batch_accuracies = []
    if rq_type == 'abs':
        for batch in batch_results.keys():
            batch_accuracies.append(sum([1 for x in batch_results[batch] if(x[1] == target_label_id) == (
                ground_truth[x[0]] == target_label_id)]) / len(batch_results[batch]))
        base_acc = sum([1 for x in orig_results.keys() if (orig_results[x] == target_label_id)
                       == (ground_truth[x] == target_label_id)])/len(orig_results.keys())
        print("--------------------------------------------")
        print("Verifying Absolute Requirement: ")
        conf_abs, mu, sigma, satisfied = calculate_confidence(batch_accuracies, base_acc, base_acc)  # abs
        print("--------------------------------------------")
        return conf_abs, mu, sigma, satisfied
    elif rq_type == 'rel':
        batch_preserved = []
        for batch in batch_results.keys():
            print(batch_results[batch])
            batch_preserved.append(sum([1 for x in batch_results[batch] if (x[1] == target_label_id) == (
                orig_results[x[0]] == target_label_id)])/len(batch_results[batch]))
        base_acc = sum([1 for x in orig_results.keys() if (orig_results[x] == target_label_id)
                       == (ground_truth[x] == target_label_id)])/len(orig_results.keys())
        print("--------------------------------------------")
        print(" Verifying Relative Requirement:")
        conf_rel, mu, sigma, satisfied = calculate_confidence(batch_preserved, base_acc, a)  # rel
        print("--------------------------------------------")

        return conf_rel, mu, sigma, satisfied
    else:
        raise ValueError("Invalid rq_type")
