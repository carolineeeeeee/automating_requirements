import sys
import scipy
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Union
from .constant import ACCURACY_PRESERVATION, PREDICTION_PRESERVATION
from .utils import get_model
from .dataset import Cifar10Dataset, ImagenetDataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(filename)s | %(lineno)d | %(message)s',
                              '%m-%d-%Y %H:%M:%S')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


def run_model(model_name: str, bootstrap_df: pd.DataFrame, cpu: bool = False, batch_size: int = 10,
              dataset_class: Union[Cifar10Dataset, ImagenetDataset] = Cifar10Dataset):
    model = get_model(model_name, pretrained=True, val=True)
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    logger.info(f"Device: {str(device)}")
    model.to(device)
    batches = bootstrap_df['bootstrap_iter_id'].unique()
    pbar = tqdm(total=len(bootstrap_df))
    prediction_records = []
    err_top_5 = 0
    err_top_1 = 0
    total_image = 0
    for bootstrap_iter_id in batches:
        df = bootstrap_df[bootstrap_df['bootstrap_iter_id'] == bootstrap_iter_id]
        dataset = dataset_class(df)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=1)

        for i, data in enumerate(dataloader):
            original_images = data['original_image'].to(device)
            new_images = data['new_image'].to(device)

            labels = data['label'].to(device)
            transformed_result = model(new_images)
            original_result = model(original_images)

            new_pred = torch.argmax(transformed_result, dim=1)
            original_pred = torch.argmax(original_result, dim=1)

            actual_batch_size = len(new_pred)
            top_1_error_count = actual_batch_size - int(torch.sum(labels == new_pred))
            top_5_error_count = actual_batch_size - int(
                torch.sum(torch.sum(torch.eq(labels.unsqueeze(1).repeat(1, 5), new_pred.topk(5).indices),
                                    dim=1)))
            total_image += actual_batch_size
            new_pred = new_pred.tolist()
            original_pred = original_pred.tolist()
            for k, label in enumerate(labels.tolist()):
                prediction_records.append({
                    'model_name': model_name,
                    'bootstrap_iter_id': bootstrap_iter_id,
                    'dataload_itr_id': i,
                    'within_iter_id': k,
                    'label': label,
                    'new_prediction': new_pred[k],
                    'original_prediction': original_pred[k],
                    'transformation': data['transformation'][k],
                    'original_filename': data['original_filename'][k],
                    'new_filename': data['new_filename'][k],
                    'original_path': data['original_path'][k],
                    'new_path': data['new_path'][k],
                    'vd_score': float(data['vd_score'][k]),
                })
            pbar.set_postfix({f'Iteration': bootstrap_iter_id})
            pbar.update(actual_batch_size)
    records_df = pd.DataFrame(data=prediction_records)
    return records_df


def calculate_confidence(acc_list, base_acc, req_acc):
    # fitting a truncated normal distribution
    def func(p, r, xa, xb):
        return scipy.stats.truncnorm.nnlf(p, r)

    def constraint(p, r, xa, xb):
        a, b, loc, scale = p
        return np.array([a*scale + loc - xa, b*scale + loc - xb])
    xa, xb = 0, 1
    mu, sigma = scipy.stats.norm.fit(acc_list)
    xa, xb = 0, 1
    mu_guess = mu
    sigma_guess = sigma
    a_guess = (xa - mu_guess)/sigma_guess
    b_guess = (xb - mu_guess)/sigma_guess
    p0 = [a_guess, b_guess, mu_guess, sigma_guess]
    par = scipy.optimize.fmin_slsqp(func, p0, f_eqcons=constraint, args=(acc_list, xa, xb),
                                    iprint=False, iter=500)
    # print(par)
    xmin = min(acc_list)
    xmax = max(acc_list)
    x = np.linspace(xmin, xmax, 1000)
    distribution = scipy.stats.truncnorm(*par)
    result = 1 - distribution.cdf(req_acc)
    fig, ax = plt.subplots(1, 1)
    # ax.plot(x, truncnorm.pdf(x, a, b, loc=loc, scale=scale),
    #        'r-', lw=3, alpha=0.4, label='truncnorm pdf')
    ax.plot(x, scipy.stats.truncnorm.pdf(x, *par),
            'k--', lw=1, alpha=1.0, label='truncnorm fit')
    ax.plot(x, scipy.stats.norm.pdf(x, mu, sigma), lw=1, alpha=1.0, label='norm fit')
    ax.hist(acc_list, bins=20, density=True, histtype='stepfilled', alpha=0.3)
    ax.legend(shadow=True)
    plt.xlim(xmin, xmax)
    plt.grid(True)
    plt.show()
    return result, par[2], par[3], result >= 0.95


'''
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
'''


def estimate_conf_int(model_df: pd.DataFrame, rq_type: str, target_label_id: int, ground_truth: Dict, a: float = 0.95):
    """
    model_df should have the following columns:
    - model_name
    - bootstrap_iter_id
    - dataload_itr_id
    - image_idx_in_batch
    - label
    - new_prediction
    - original_prediction
    - transformation
    - original_filename
    - new_filename
    - original_path
    - new_path
    """
    # model_df
    model_df.drop_duplicates(subset=['original_filename']).set_index('original_filename')
    orig_results = {row['original_filename']: row['original_prediction'] for i, row in model_df.iterrows()}
    batch_ids = model_df['bootstrap_iter_id'].unique()
    batch_results = {batch_id: [(row['original_filename'], row['new_prediction'])
                                for i, row in model_df[model_df['bootstrap_iter_id'] == batch_id].iterrows()] for
                     batch_id in batch_ids}
    batch_accuracies = []
    if rq_type == ACCURACY_PRESERVATION:
        for batch in batch_results.keys():
            batch_accuracies.append(sum([1 for x in batch_results[batch] if (x[1] == target_label_id) == (
                ground_truth[x[0]] == target_label_id)]) / len(batch_results[batch]))
        base_acc = sum([1 for x in orig_results.keys() if (orig_results[x] == target_label_id)
                        == (ground_truth[x] == target_label_id)]) / len(orig_results.keys())
        print("--------------------------------------------")
        print("Verifying Absolute Requirement: ")
        conf_abs, mu, sigma, satisfied = calculate_confidence(batch_accuracies, base_acc, base_acc)  # abs
        print("--------------------------------------------")
        return conf_abs, mu, sigma, satisfied
    elif rq_type == PREDICTION_PRESERVATION:
        batch_preserved = []
        for batch in batch_results.keys():
            # print(batch_results[batch])
            batch_preserved.append(sum([1 for x in batch_results[batch] if (x[1] == target_label_id) == (
                orig_results[x[0]] == target_label_id)]) / len(batch_results[batch]))
        base_acc = sum([1 for x in orig_results.keys() if (orig_results[x] == target_label_id)
                        == (ground_truth[x] == target_label_id)]) / len(orig_results.keys())
        print("--------------------------------------------")
        print(" Verifying Relative Requirement:")
        conf_rel, mu, sigma, satisfied = calculate_confidence(batch_preserved, base_acc, a)  # rel
        print("--------------------------------------------")

        return conf_rel, mu, sigma, satisfied
    else:
        raise ValueError("Invalid rq_type")


def obtain_preserved_min_degradation(record_df):
    # compare boot_df filename and record_df transformed_filename
    # find percenrage preserved within minimum IQA range, return it
    range_limit = np.percentile(np.array(record_df['vd_score']), 10)
    #range_limit = min(record_df['vd_score']) + MIN_IQA_RANGE
    min_range_predictions = record_df.loc[record_df['vd_score'] <= range_limit]
    predictions_preserved = record_df.loc[((record_df['original_prediction'] == 1) == (
        record_df['new_prediction'] == 1)) & (record_df['vd_score'] <= range_limit)]
    return len(predictions_preserved) / float(len(min_range_predictions))
