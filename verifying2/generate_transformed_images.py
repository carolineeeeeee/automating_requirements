import torch
import pathlib2
import torchvision
from src import constant
from src.utils import get_transformation_threshold, transform_image_dir
from src.utils import start_matlab

transformation_type = constant.CONTRAST

threshold = get_transformation_threshold(transformation_type, 'abs')
matlab_eng = start_matlab(constant.IQA_PATH, constant.matlabPyrToolsPath)

for dataset_type in ['val', 'train']:
    dataset = constant.ROOT / f"data/cifar10_pytorch/{dataset_type}"
    target = f"./transformed_data/{dataset_type}"
    new_labels = []
    transformed_paths = transform_image_dir(
        matlab_eng, threshold, pathlib2.Path(dataset),
        pathlib2.Path(target),
        transformation_type=transformation_type)
