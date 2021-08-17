import os
import glob
import random
import shutil
import cv2
import pathlib2
import numpy as np
import pandas as pd
import matlab.engine
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from typing import List
from typing import Union, Dict
from robustbench import load_model
import torchvision.models as models

from .Imagenet_c_transformations import *
from .constant import GAUSSIAN_NOISE, SHOT_NOISE, IMPULSE_NOISE, MOTION_BLUR, SNOW, FROST, FOG, BRIGHTNESS, CONTRAST, \
    JPEG_COMPRESSION, \
    ALEXNET, \
    GOOGLENET, \
    RESNET50, \
    RESNEXT50, \
    VGG_16, \
    THRESHOLD_MAP, \
    ROBUSTBENCH_CIFAR10_MODEL_NAMES, \
    GENERALIZED, \
    CONTRAST_G, \
    UNIFORM_NOISE, \
    LOWPASS, \
    HIGHPASS, \
    PHASE_NOISE, \
    DEFOCUS_BLUR, \
    GLASS_BLUR


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
    Read labels information from a dataset, 2 types of files are accepted, labels.txt and labels.csv.
    In the case where images are labelled strictly in numeric order (continuous) (1.png, 2.png, ..., 100.png), 
    labels.txt is just a list of labels (no need to save filenames as it's in order minimizing the space).
    In the case filenames are not continuous or filenames are not integers (2.png, transformed_100.png, ...) a csv file
    called labels.csv can be used to map filenames and labels.
    The first method (labels.txt) may be deprecated to keep this function simpler and robust.
    :param data_path: directory containing all images and a labels.txt or a labels.csv
    :return:
    """
    if (data_path / 'labels.txt').exists():
        labels = np.loadtxt(str(data_path / 'labels.txt')).astype(int)
        indices = list(range(len(labels)))
        df = pd.DataFrame(data={"index": indices, "label": labels})
        df['original_filename'] = df['index'].apply(lambda i: f"{i}.png")
    elif (data_path / 'labels.csv').exists():
        df = pd.read_csv(data_path / 'labels.csv', index_col=0)   # should contain 2 columns: filename, label
        df['original_filename'] = df['filename']
    else:
        raise OSError(f"label file not found, either labels.txt or labels.csv should be present in {data_path}")
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


def transform_image(image_path: str, transformation: str):
    if transformation in [
            GAUSSIAN_NOISE, SHOT_NOISE, IMPULSE_NOISE, MOTION_BLUR, SNOW, FROST, FOG, BRIGHTNESS, CONTRAST,
            JPEG_COMPRESSION]:
        img = Image.open(image_path)
    else:
        img = np.asarray(cv2.imread(image_path), dtype=np.float32)
    # ============= different transformation types begin =============
    if transformation == CONTRAST_G:
        param = random.choice(contrast_params)
        img2 = adjust_contrast(img, param)
    elif transformation == UNIFORM_NOISE:
        param = random.choice(uniform_noise_params)
        img2 = apply_uniform_noise(img, 0, param)
    elif transformation == LOWPASS:
        param = random.choice(lowpass_params)
        img2 = low_pass_filter(img, param)
    elif transformation == HIGHPASS:
        param = random.choice(highpass_params)
        img2 = high_pass_filter(img, param)
    elif transformation == PHASE_NOISE:
        param = random.choice(phase_noise_params)
        img2 = scramble_phases(img, param)
    elif transformation == GAUSSIAN_NOISE:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, param = gaussian_noise(img, param_index)
    elif transformation == SHOT_NOISE:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, param = shot_noise(img, param_index)
    elif transformation == IMPULSE_NOISE:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, param = impulse_noise(img, param_index)
    elif transformation == DEFOCUS_BLUR:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, param = defocus_blur(img, param_index)
    elif transformation == GLASS_BLUR:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, param = glass_blur(img, param_index)
    elif transformation == MOTION_BLUR:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, param = motion_blur(img, param_index)
    elif transformation == SNOW:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2 = snow(img, param_index)
    elif transformation == FROST:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, _ = frost(img, param_index)
    elif transformation == FOG:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, _ = fog(img, param_index)
    elif transformation == BRIGHTNESS:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, _ = brightness(img, param_index)
    elif transformation == CONTRAST:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, _ = contrast(img, param_index)
    elif transformation == JPEG_COMPRESSION:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, _ = jpeg_compression(img, param_index)
        img2 = np.asarray(img2)
        # ============= different transformation types end =============
    else:
        raise ValueError("Invalid Transformation")
    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
    return img2, img2_g


def transform_image_dir(
        matlab_engine, threshold: float, source_dir: pathlib2.Path, target_dir: pathlib2.Path, transformation_type: str,
        accepted_ext: Union[List, str] = ['png'], max_num_trial: int = 10) -> List[str]:
    if isinstance(accepted_ext, str):
        accepted_ext = [accepted_ext]
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    labels_txt_path = os.path.join(source_dir, 'labels.txt')
    labels_data = np.loadtxt(labels_txt_path).astype(np.int)
    new_filenames = []
    new_labels = []
    transformed_image_paths = []
    count = 0
    for ext in accepted_ext:
        for image_path in tqdm(glob.glob(f'{source_dir}/*.{ext}')):
            img_g = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # greyscale image
            for _ in range(max_num_trial):
                transformed_image, _ = transform_image(image_path, transformation_type)
                img2_g = cv2.cvtColor(
                    np.float32(transformed_image),
                    cv2.COLOR_BGR2GRAY) if len(
                    transformed_image.shape) == 3 else transformed_image
                try:
                    IQA_score = matlab_engine.vifvec2_layers(
                        matlab.double(np.asarray(img_g).tolist()),
                        matlab.double(np.asarray(img2_g).tolist()))
                except Exception as e:
                    logger.error("failed, try again")
                    continue
                if 1 - IQA_score < threshold:
                    # path to save transformed image
                    image_name = os.path.basename(image_path)
                    label = labels_data[int(image_name.split('.')[0])]
                    new_labels.append(label)
                    new_name = f'{transformation_type}_{image_name}'
                    new_filenames.append(new_name)
                    target_path = str(target_dir / new_name)
                    cv2.imwrite(target_path, transformed_image)
                    transformed_image_paths.append(target_path)
                    count += 1
                    break
    df = pd.DataFrame(data={
        'filename': new_filenames,
        'label': new_labels
    })
    df.to_csv(os.path.join(target_dir, 'labels.csv'))
    return transformed_image_paths
