import os
import numpy as np
import torch.nn as nn
import pathlib2
from typing import Union
import pandas as pd
from typing import Dict
import cv2
from .constant import ALEXNET, \
    GOOGLENET, \
    RESNET50, \
    RESNEXT50, \
    VGG_16, \
    TRANSFORMATIONS
# TRANSFORMATION_PARAM_MAP_DOMAIN
import torchvision.models as models
from robustbench.utils import load_model


def get_model(model_name: str, pretrained: bool = True, val: bool = True) -> nn.Module:
    """Return a pretrained model give a model name

    :param model_name: model name, choose from [alexnet, googlenet, resnet50, resnext50, vgg-16]
    :type model_name: str
    :param pretrained: Whether the model should be pretrained, defaults to True
    :type pretrained: bool, optional
    :param val: Whether the model in validation model (prevent weight from updating), defaults to True
    :type val: bool, optional
    :raises ValueError: Invalid Model Name
    :raises ValueError: Invalid Model Name
    :return: a pytorch model
    :rtype: nn.Module
    """
    if model_name == ALEXNET:
        model = models.alexnet(pretrained=pretrained)
    elif model_name == GOOGLENET:
        model = models.googlenet(pretrained=pretrained)
    elif model_name == RESNET50:
        model = models.resnet50(pretrained=pretrained)
    elif model_name == RESNEXT50:
        model = models.resnext50_32x4d(pretrained=pretrained)
    elif model_name == VGG_16:
        model = models.vgg16(pretrained=pretrained)
    elif model_name in ['Standard', 'Engstrom2019Robustness', 'Rice2020Overfitting', 'Carmon2019Unlabeled']:
        model = load_model(model_name)
    else:
        raise ValueError("Invalid Model Name")
    if val:
        model.eval()
    return model


# def get_random_parameter_for_transformation(transformation_type: str) -> float:
#     """
#     Get a random transformation parameter given a transformation type
#     :param transformation_type: the transformation type to perform on an image
#     :type transformation_type: str
#     :return: transformation parameter
#     :rtype: float
#     """
#     if transformation_type not in TRANSFORMATIONS:
#         raise ValueError(f"transformation_type invalid. Has to be picked from {TRANSFORMATIONS}.")
#     param_domain = TRANSFORMATION_PARAM_MAP_DOMAIN[transformation_type]
#     return np.random.uniform(param_domain[0], param_domain[1])


def dir_is_empty(path: Union[str, pathlib2.Path]) -> bool:
    if os.path.exists(path):
        return len(os.listdir(path)) == 0
    else:
        raise OSError("Path doesn't exist")


def load_image_data(root: pathlib2.Path, category: str):
    data_dir = root / 'cifar10_data' / category
    labels = np.loadtxt(data_dir/'labels.txt').astype(int)
    indices = list(range(len(labels)))
    df = pd.DataFrame(data={"index": indices, "label": labels})
    df['filename'] = df['index'].apply(lambda i: f"{i}.png")
    df['path'] = df['filename'].apply(lambda filename: os.path.join(data_dir, filename))
    return df.drop(columns='index')


def bootstrap_save_record(
        batch_id: int, within_batch_id: int, original_filename: str, filename: str, original_path: str,
        transformation: str, transformed_path: str, label: int, img: np.ndarray, bootstrap_data: Dict):
    cv2.imwrite(transformed_path, img)
    bootstrap_data['batch_id'].append(batch_id)
    bootstrap_data['within_batch_id'].append(within_batch_id)
    bootstrap_data['original_filename'].append(original_filename)
    bootstrap_data['filename'].append(filename)
    bootstrap_data['original_path'].append(original_path)
    bootstrap_data['transformation'].append(transformation)
    bootstrap_data['transformed_path'].append(transformed_path)
    bootstrap_data['label'].append(label)


def read_cifar10_ground_truth(filepath: str, use_filename: bool = True):
    loaded_file = np.loadtxt(filepath).astype(int)
    if use_filename:
        ground_truth = {f"{i}.png": val for i, val in enumerate(loaded_file)}
    else:
        ground_truth = {i: val for i, val in enumerate(loaded_file)}
    return ground_truth
