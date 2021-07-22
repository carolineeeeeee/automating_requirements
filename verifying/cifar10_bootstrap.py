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
from cifar10 import CustomCIFAR10Dataset
from src.constant import TRANSFORMATIONS, \
    INTENSITY_SHIFT, \
    OBJECT_CLASSES, \
    INDEX_TO_LABEL_ID_JSON, \
    INDEX_TO_LABELS_JSON, \
    CLASS_TO_LABEL_ID_JSON, \
    IMAGE_TO_LABEL_ID_CSV, \
    CIFAR10_MODELS, \
    MODELS, \
    IMAGENET_DEFAULT_TRANSFORMATION, \
    CIFAR10_DEFAULT_TRANSFORMATION, \
    CIFAR10_CLASSES
from src.dataset import ImagenetDataset
from src.helper import get_random_parameter_for_transformation, get_model

__root__ = pathlib2.Path(__file__).absolute().parent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
file_handler = logging.FileHandler('new_bootstrap.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)


def calculate_confidence(acc_list: List[float], base_acc: float, req_acc: float) -> Tuple[float]:
    """Calculate Confidence

    :param acc_list: [description]
    :type acc_list: List[float]
    :param base_acc: [description]
    :type base_acc: float
    :param req_acc: [description]
    :type req_acc: float
    :return: confidence, mu (mean), sigma
    :rtype: Tuple[float]
    """
    # fitting a normal distribution
    # print(acc_list)
    _, bins, _ = plt.hist(acc_list, 30, alpha=0.5, density=True)
    mu, sigma = scipy.stats.norm.fit(data=acc_list)
    distribution = scipy.stats.norm(mu, sigma)
    result = 1 - distribution.cdf(req_acc)
    return round(result, 3), round(mu, 3), round(sigma, 3)


def estimate_confidence_interval(model_df: pd.DataFrame, label_map: Dict, target_label_id: int, a: int = 1) -> Tuple:
    # make a dictionary that maps a image name to its original image prediction
    model_df_dict = model_df.drop_duplicates(subset=['dataset_index']).set_index('dataset_index').to_dict("index")
    original_pred_dict = {index: label_map[model_df_dict[index]['original_prediction']]['label_id'] for index
                          in model_df_dict.keys()}
    # sort batch results
    batch_results_abs = defaultdict(dict)
    for i, record in model_df.iterrows():
        batch_results_abs[record['bootstrap_itr_id']][record['dataset_index']] = label_map[
            record['transformed_prediction']]['label_id']
    batch_results_rel = batch_results_abs.copy()

    # obtain abs
    batch_accuracies = []
    for batch_itr in batch_results_abs.keys():
        if len(batch_results_abs[batch_itr].keys()) > 0:
            s = 0
            for index in batch_results_abs[batch_itr].keys():
                if (batch_results_abs[batch_itr][index] == target_label_id) == (
                        label_map[index]['label_id'] == target_label_id):
                    s += 1
            batch_accuracies.append(s / len(batch_results_abs[batch_itr].keys()))

    batch_preserved = []
    for batch_itr in batch_results_rel.keys():
        if len(batch_results_rel[batch_itr].keys()) > 0:
            s = 0
            for index in batch_results_rel[batch_itr].keys():
                if (batch_results_rel[batch_itr][index] == target_label_id) == (
                        label_map[model_df.loc[(model_df['bootstrap_itr_id'] == batch_itr) & (
                                model_df['dataset_index'] == index)].iloc[0][
                            'original_prediction']]['label_id'] == target_label_id):
                    s += 1
            batch_preserved.append(s / len(batch_results_rel[batch_itr].keys()))

    base_acc = sum(
        [1 for index in original_pred_dict.keys() if (original_pred_dict[index] == target_label_id)
         == (label_map[index]['label_id'] == target_label_id)]) / len(model_df)
    conf_abs, mu_abs, sigma_abs = calculate_confidence(batch_accuracies, base_acc, base_acc)
    conf_rel, mu_rel, sigma_rel = calculate_confidence(batch_preserved, base_acc, a)
    return conf_abs, mu_abs, sigma_abs, conf_rel, mu_rel, sigma_rel


def generate_bootstrap_payload(
        label_df: pd.DataFrame, num_bootstrap_batches: int, object_class: str, bootstrap_batch_size: int = 50,
        transformation_type: TRANSFORMATIONS = INTENSITY_SHIFT) -> List[
    pd.DataFrame]:
    """[summary]

    :param label_df: [description]
    :type label_df: pd.DataFrame
    :param num_bootstrap_batches: [description]
    :type num_bootstrap_batches: int
    :param object_class: [description]
    :type object_class: str
    :param bootstrap_batch_size: [description], defaults to 50
    :type bootstrap_batch_size: int, optional
    :param transformation_type: [description], defaults to INTENSITY_SHIFT
    :type transformation_type: TRANSFORMATIONS, optional
    :return: [description]
    :rtype: List[ pd.DataFrame]
    """
    logger.info("Generating Bootstrap Parameters")
    is_object_class_indicators = label_df['label'] == object_class
    is_object_df = label_df[is_object_class_indicators]
    not_object_df = label_df[~is_object_class_indicators]
    target_image_df = pd.concat(
        [is_object_df, not_object_df.sample(n=bootstrap_batch_size, replace=False, random_state=1)])

    transform_dfs = []
    for i in tqdm(range(num_bootstrap_batches)):
        sample_df = target_image_df.sample(n=bootstrap_batch_size, replace=False, random_state=i)
        sample_df['transformation_type'] = transformation_type
        sample_df['transformation_param'] = [get_random_parameter_for_transformation(
            transformation_type) for _ in range(bootstrap_batch_size)]
        transform_dfs.append(sample_df)
    return transform_dfs


def get_ml_metric(model_df: pd.DataFrame, target_label_id: int) -> Tuple:
    """[summary]

    :param model_df: table/dataframe containing prediction data of all images (predicted by a specific model)
    :type model_df: pd.DataFrame
    :return: [description]
    :rtype: Tuple
    """
    true_pos, true_neg, false_pos, false_neg = np.int32(0), np.int32(0), np.int32(0), np.int32(0)
    for i, record in model_df.iterrows():
        # original_label_id = index_to_label_id[record['original_prediction']]
        # transformed_label_id = index_to_label_id[record['transformed_prediction']]
        # target_label_id = record['label_id']
        if target_label_id == record['label_index']:
            if record['original_prediction'] == record['label_index']:  # true positive
                true_pos += 1
            else:  # false negative
                false_neg += 1
            if record['transformed_prediction'] == record['label_index']:  # true positive
                true_pos += 1
            else:  # false negative
                false_neg += 1
        else:
            if record['original_prediction'] == record['label_index']:  # false positive
                false_pos += 1
            else:  # true negative
                true_neg += 1
            if record['transformed_prediction'] == record['label_index']:  # false positive
                false_pos += 1
            else:  # true negative
                true_neg += 1
    accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg), 3)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = round(2 * (recall * precision) / (recall + precision), 3)
    return accuracy, f1


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Argument Parser Setup
    Add argument parsing rules to the parser
    :param parser: argument parser object
    :type parser: argparse.ArgumentParser
    :return: args parsed from command line arguments
    :rtype: argparse.Namespace
    """
    parser.add_argument('-n', "--number_of_batches", type=int, default=200,
                        help='the number of batches for bootstrapping, must be integer')
    parser.add_argument('-s', "--sample_size", type=int, default=500, help="bootstrap sample size. "
                                                                           "the number of images per batch for "
                                                                           "bootstrapping, must be integer")
    parser.add_argument('--batch_size', type=int, default=10,
                        help='number of images passed to network at once. Large batch_size takes more RAM or GPU Mem')
    parser.add_argument('-c', '--class', type=str, default="car",
                        help="the object class y to test in both requirements", choices=OBJECT_CLASSES)
    parser.add_argument('--cpu', action='store_true', help='use cpu for computation even when a GPU is available')
    parser.add_argument('-t', '--transformation', type=str, default=INTENSITY_SHIFT, choices=TRANSFORMATIONS,
                        help='the transformation to test')
    return parser.parse_args()


def main(args_payload: Dict) -> None:
    """main function/Entrypoint to this program

    :param args_payload: dictionary mapping command line argument keywords to actual argument values, args.__dict__
    :type args_payload: Dict
    """
    logger.info("Arguments:")
    for key, value in args_payload.items():
        logger.info(f"\t{key}: {value}")
    # general setup
    device = torch.device('cuda' if torch.cuda.is_available() and not args_payload['cpu'] else 'cpu')
    logger.info(f"device: {str(device)}")
    if torch.cuda.is_available():
        logger.info(f"device name: {str(torch.cuda.get_device_name(torch.cuda.current_device()))}")
    # reading static static data files
    logger.info("reading cifar10 info")
    cifar10_test_label_df = pd.read_csv(os.path.join(__root__, "static", "cifar10_test_label.csv"), index_col=0)
    cifar10_test_label_dict = cifar10_test_label_df.to_dict('index')
    logger.info("generate bootstrap transformation plan")
    bootstrap_payload = generate_bootstrap_payload(
        cifar10_test_label_df,
        args_payload["number_of_batches"],
        args_payload['class'],
        args_payload["sample_size"],
        args_payload["transformation"])

    logger.info("Start Evaluating Models")
    total_num_images = sum([len(df) for df in bootstrap_payload])
    logger.debug(f"total number of image to run: {total_num_images}")
    performance_df = pd.DataFrame(0, columns=['err_count@1', 'err_count@5', 'total', 'err@1', 'err@5'],
                                  index=CIFAR10_MODELS)
    progress_bar = tqdm(total=len(CIFAR10_MODELS) * total_num_images)
    prediction_records = []
    
    for model_name in CIFAR10_MODELS:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = get_model(model_name, pretrained=True, val=True)
        model.to(device)

        for i, bootstrap_df in enumerate(bootstrap_payload):
            dataset = CustomCIFAR10Dataset(bootstrap_df, root='./data', train=False, download=True,
                                           transform=CIFAR10_DEFAULT_TRANSFORMATION)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args_payload['batch_size'],
                                                     shuffle=False, num_workers=2)
            for j, data in enumerate(dataloader):
                original_images = data['original_image'].to(device)
                transformed_images = data['transformed_image'].to(device)
                target = data['label_index'].to(device)
                transformation_param = data['transformation_param'].tolist()
                transformed_result = model(transformed_images)
                original_result = model(original_images)
                transformed_pred = torch.argmax(transformed_result, dim=1)
                original_pred = torch.argmax(original_result, dim=1)
                actual_batch_size = len(transformed_pred)
                top_1_error_count = actual_batch_size - int(torch.sum(target == transformed_pred))
                top_5_error_count = actual_batch_size - int(
                    torch.sum(torch.sum(torch.eq(target.unsqueeze(1).repeat(1, 5), transformed_result.topk(5).indices),
                                        dim=1)))
                performance_df.loc[model_name, 'err_count@1'] += top_1_error_count
                performance_df.loc[model_name, 'err_count@5'] += top_5_error_count
                performance_df.loc[model_name, 'total'] += actual_batch_size
                # obtain original prediction label
                # original_pred_labels = [index_to_labels[pred_idx] for pred_idx in original_pred.tolist()]
                # original_first_pred_label = [pred_labels[0] for pred_labels in original_pred_labels]
                # obtain transformed prediction label
                # transformed_pred_labels = [index_to_labels[pred_idx] for pred_idx in transformed_pred.tolist()]
                # transformed_first_pred_label = [pred_labels[0] for pred_labels in transformed_pred_labels]
                transformed_pred, original_pred = transformed_pred.tolist(), original_pred.tolist()
                for k, target in enumerate(target.tolist()):
                    prediction_records.append({
                        'model_name': model_name,
                        'bootstrap_itr_id': i,
                        'dataload_itr_id': j,
                        'image_idx_in_batch': k,
                        'label_index': target,
                        'dataset_index': int(data['dataset_index'][k]),
                        'transformed_prediction': transformed_pred[k],
                        'original_prediction': original_pred[k],
                        'transformation_type': data['transformation_type'][k],
                        'transformation_param': transformation_param[k],
                        # 'original_pred_labels': original_pred_labels[k],
                        # 'original_first_pred_label': original_first_pred_label[k],
                        # 'transformed_pred_labels': transformed_pred_labels[k],
                        # 'transformed_first_pred_label': transformed_first_pred_label[k]
                    })
                progress_bar.update(len(transformed_images))
                progress_bar.set_postfix({'model': model_name})
    records_df = pd.DataFrame(data=prediction_records)
    records_df.to_csv("records.csv")
    # analysis
    metric_df = pd.DataFrame(columns=["mu1", "sigma1", "conf_acc1", "mu2", "sigma2", "conf_acc2", "accuracy", "f1"],
                             index=CIFAR10_MODELS)
    target_label_id = CIFAR10_CLASSES.index(args_payload['class'])
    for model_name in CIFAR10_MODELS:
        model_df = records_df[records_df['model_name'] == model_name]
        # obj_cls_label_id_set = set(class_to_label_id_dict[args_payload['class']])
        accuracy, f1 = get_ml_metric(model_df, target_label_id)
        conf_abs, mu_abs, sigma_abs, conf_rel, mu_rel, sigma_rel = estimate_confidence_interval(model_df,
                                                                                                cifar10_test_label_dict,
                                                                                                target_label_id)
        metric_df.loc[model_name] = [mu_abs, sigma_abs, conf_abs, mu_rel, sigma_rel, conf_rel, accuracy, f1]
    logger.info("Metrics:\n" + tabulate(metric_df, headers='keys', tablefmt='pretty'))
    performance_df['err@1'] = performance_df['err_count@1'] / performance_df['total']
    performance_df['err@5'] = performance_df['err_count@5'] / performance_df['total']
    performance_df['acc@1'] = 1 - performance_df['err@1']
    performance_df['acc@5'] = 1 - performance_df['err@5']
    logger.info("Accuracy Performance:\n" + tabulate(performance_df.round(3), headers='keys', tablefmt='pretty'))
    performance_df.round(3).to_csv(str(__root__ / "bootstrap_performance.csv"))

    logger.info("Correlations:")
    abs_acc, _ = pearsonr(metric_df['conf_acc1'], metric_df['accuracy'])
    logger.info("conf_acc and accuacy of test images: " + str(round(abs_acc, 3)))
    abs_f1, _ = pearsonr(metric_df['conf_acc1'], metric_df['f1'])
    logger.info("conf_acc and f1 of test images: " + str(round(abs_f1, 3)))
    rel_acc, _ = pearsonr(metric_df['conf_acc2'], metric_df['accuracy'])
    logger.info("conf_pred and accuracy of test images: " + str(round(rel_acc, 3)))
    rel_f1, _ = pearsonr(metric_df['conf_acc2'], metric_df['f1'])
    logger.info("conf_pred and f1 of test images: " + str(round(rel_f1, 3)))
    abs_rel, _ = pearsonr(metric_df['conf_acc1'], metric_df['conf_acc2'])
    logger.info("conf_acc and conf_pred: " + str(round(abs_rel, 3)))
    logger.info("main function finished")


if __name__ == "__main__":
    main(parse_args(
        argparse.ArgumentParser("This is the script for our subprocess IV Verifying MLC against requirement")).__dict__)
    logger.info("__main__ function finished")
