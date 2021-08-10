import os
import shutil

import cv2
import pathlib2
import numpy as np
import pandas as pd
import matlab.engine
import torch.nn as nn
from typing import Union, Dict

import torchvision.models as models

from .constant import ALEXNET, \
    GOOGLENET, \
    RESNET50, \
    RESNEXT50, \
    VGG_16, \
    TRANSFORMATIONS, \
    THRESHOLD_MAP, \
    ROBUSTBENCH_CIFAR10_MODEL_NAMES, \
    GENERALIZED

import torchvision.models as models
from robustbench import load_model


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
    elif model_name in ROBUSTBENCH_CIFAR10_MODEL_NAMES:
        if 'L2' in model_name:
            model = load_model(model_name=model_name, dataset='cifar10', threat_model="L2")
        elif 'Linf' in model_name:
            model = load_model(model_name=model_name, dataset='cifar10', threat_model="Linf")
        model = load_model(model_name=model_name, dataset='cifar10', threat_model="corruptions")
    else:
        raise ValueError(f"Invalid Model Name: {model_name}")
    if val:
        model.eval()
    return model


def dir_is_empty(path: Union[str, pathlib2.Path]) -> bool:
    if os.path.exists(path):
        return len(os.listdir(path)) == 0
    else:
        raise OSError("Path doesn't exist")


def get_transformation_threshold(transformation: str, rq_type: str):
    return THRESHOLD_MAP[transformation][rq_type] if transformation in THRESHOLD_MAP else THRESHOLD_MAP[GENERALIZED][
        rq_type]


def read_cifar10_ground_truth(label_path: str, use_filename: bool = True):
    """
    :param label_path:
    :param use_filename: whether using filename as the key or number as the key
    :return:
    """
    loaded_file = np.loadtxt(label_path).astype(int)
    if use_filename:
        # use filename as the key
        ground_truth = {f"{i}.png": val for i, val in enumerate(loaded_file)}
    else:
        # use the index number in image name as the key
        ground_truth = {i: val for i, val in enumerate(loaded_file)}
    return ground_truth


def load_cifar10_data(data_path: pathlib2.Path):
    """
    :param data_path: directory containing all images and a labels.txt
    :return:
    """
    labels = np.loadtxt(str(data_path / 'labels.txt')).astype(int)
    indices = list(range(len(labels)))
    df = pd.DataFrame(data={"index": indices, "label": labels})
    df['original_filename'] = df['index'].apply(lambda i: f"{i}.png")
    df['original_path'] = df['original_filename'].apply(lambda filename: os.path.join(str(data_path), filename))
    return df.drop(columns='index')


def load_imagenet_data(data_path: pathlib2.Path, image_to_label_id_csv_path: pathlib2.Path):
    image_to_label_id_df = pd.read_csv(image_to_label_id_csv_path, index_col=0)
    label_map_dict = image_to_label_id_df.to_dict('index')
    lst = [{
            'original_path': str(path),
            'original_filename': path.name,
            'label': label_map_dict[path.name.split('.')[0]]['label_index']
        } for path in data_path.iterdir()]
    return pd.DataFrame(data=lst)


def start_matlab(IQA_PATH: str, matlabPyrToolsPath: str):
    eng = matlab.engine.start_matlab()
    eng.addpath(IQA_PATH, nargout=0)
    eng.addpath(matlabPyrToolsPath, nargout=0)
    eng.addpath(matlabPyrToolsPath + '/MEX', nargout=0)
    return eng


def clean_dir(path: Union[str, pathlib2.Path]):
    path = pathlib2.Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def dict_to_str(d: Dict) -> str:
    return "\n".join([f"{key}: {value}" for key, value in d.items()])
