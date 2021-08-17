import os
import torch
import pathlib2
import numpy as np
import torchvision
from src import constant
from src.utils import get_transformation_threshold, transform_image_dir
from src.utils import start_matlab
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

transformation_type = constant.CONTRAST

threshold = get_transformation_threshold(transformation_type, 'abs')
matlab_eng = start_matlab(constant.IQA_PATH, constant.matlabPyrToolsPath)

for dataset_type in ['val', 'train']:
    dataset = f"/home/huakun/Documents/Work/research/automating_requirements/verifying2/data/cifar10_pytorch/{dataset_type}"
    target = f"./transformed_data/{dataset_type}"
    new_labels = []
    transformed_paths = transform_image_dir(
        matlab_eng, threshold, pathlib2.Path(dataset),
        pathlib2.Path(target),
        transformation_type=transformation_type)
