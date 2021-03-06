import glob
import torch
import shutil
import torchvision
import pandas as pd
import matlab.engine
import torch.nn as nn
from tqdm import tqdm
from typing import List
from tabulate import tabulate
from typing import Union, Dict
from robustbench import load_model
from collections import OrderedDict
import torchvision.models as models

from .Imagenet_c_transformations import *
from .constant import GAUSSIAN_NOISE, FROST, BRIGHTNESS, CONTRAST, JPEG_COMPRESSION, ALEXNET, GOOGLENET, RESNET50, \
    RESNEXT50, VGG_16, THRESHOLD_MAP, ROBUSTBENCH_CIFAR10_MODEL_NAMES, DEFOCUS_BLUR, ANT3x3_Model, ANT3x3_SIN_Model, \
    ANT_Model, ANT_SIN_Model, Gauss_mult_Model, Gauss_sigma_0, Speckle_Model, DEEPAUGMENT_AND_AUGMIX, \
    RESNEXT101_AUGMIX_AND_DEEPAUGMENT, DEEPAUGMENT, IMAGENETC_MODEL_PATH


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
    elif model_name in [ANT3x3_Model, ANT3x3_SIN_Model, ANT_Model, ANT_SIN_Model, Gauss_mult_Model, Gauss_sigma_0, Speckle_Model]:
        model = torchvision.models.resnet50(pretrained=False)
        model.load_state_dict(
            torch.load(str(IMAGENETC_MODEL_PATH / f"{model_name}.pth"))['model_state_dict'])
    elif model_name in [DEEPAUGMENT_AND_AUGMIX, DEEPAUGMENT, RESNEXT101_AUGMIX_AND_DEEPAUGMENT]:
        new_state_dict = OrderedDict()
        state_dict = torch.load(str(IMAGENETC_MODEL_PATH / f"{model_name}.pth.tar"))['state_dict']
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        if model_name == RESNEXT101_AUGMIX_AND_DEEPAUGMENT:
            model = torchvision.models.resnext101_32x8d(pretrained=False)
        else:
            model = torchvision.models.resnet50(pretrained=False)
        model.load_state_dict(new_state_dict)
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
    return THRESHOLD_MAP[transformation][rq_type]


def read_cifar10_ground_truth(label_path: str):
    """
    :param label_path:
    :param use_filename: whether using filename as the key or number as the key
    :return:
    """
    label_df = pd.read_csv(label_path, index_col=0)
    return {row['filename']: row['label'] for i, row in label_df.iterrows()}


def load_cifar10_data(data_path: pathlib2.Path):
    """
    Read labels information from a dataset
    Labels.csv is a mapping for filenames and labels.
    :param data_path: directory containing all images and a labels.csv
    :return:
    """
    if (data_path / 'labels.csv').exists():
        df = pd.read_csv(data_path / 'labels.csv', index_col=0)   # should contain 2 columns: filename, label
        df['original_filename'] = df['filename']
        df['original_path'] = df['original_filename'].apply(lambda filename: os.path.join(str(data_path), filename))
        return df
    else:
        raise OSError(f"label file not found, labels.csv should be present in {data_path}")


def load_imagenet_data(data_path: pathlib2.Path, image_to_label_id_csv_path: pathlib2.Path):
    image_to_label_id_df = pd.read_csv(image_to_label_id_csv_path, index_col=0)
    label_map_dict = image_to_label_id_df.to_dict('index')
    lst = [{
        'original_path': str(path),
        'original_filename': path.name,
        'label': label_map_dict[path.name.split('.')[0]]['label_index']
    } for path in data_path.iterdir() if path.name != "labels.csv"]
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
        shutil.rmtree(str(path))
    path.mkdir(parents=True, exist_ok=True)


def dict_to_str(d: Dict) -> str:
    return "\n".join([f"{key}: {value}" for key, value in d.items()])


def transform_image(image_path: str, transformation: str):
    if transformation in [
            GAUSSIAN_NOISE, FROST, BRIGHTNESS, CONTRAST,
            JPEG_COMPRESSION]:
        img = Image.open(image_path)
    else:
        img = np.asarray(cv2.imread(image_path), dtype=np.float32)
    # ============= different transformation types begin =============
    if transformation == GAUSSIAN_NOISE:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, param = gaussian_noise(img, param_index)
    elif transformation == DEFOCUS_BLUR:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, param = defocus_blur(img, param_index)
    elif transformation == FROST:
        param_index = random.choice(range(TRANSFORMATION_LEVEL))
        img2, _ = frost(img, param_index)
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
    label_df = pd.read_csv(source_dir/'labels.csv', index_col=0)
    label_dict = {row['filename']: row['label'] for i, row in label_df.iterrows()}
    new_filenames = []
    new_labels = []
    transformed_image_paths = []
    IQA_scores = []
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
                    label = label_dict[image_name]
                    new_labels.append(label)
                    new_name = f'{transformation_type}_{image_name}'
                    new_filenames.append(new_name)
                    IQA_scores.append(IQA_score)
                    target_path = str(target_dir / new_name)
                    cv2.imwrite(target_path, transformed_image)
                    transformed_image_paths.append(target_path)
                    count += 1
                    break
    df = pd.DataFrame(data={
        'filename': new_filenames,
        'label': new_labels,
        'IQA_score': IQA_scores
    })
    df.to_csv(os.path.join(target_dir, 'labels.csv'))
    return transformed_image_paths


def visualize_table(
        csv_path: Union[str, pathlib2.Path] = None, dataframe: pd.DataFrame = None, print_: bool = True) -> str:
    if csv_path is not None:
        df = pd.read_csv(csv_path, index_col=0)
    elif dataframe is not None:
        df = dataframe
    else:
        raise ValueError('either csv_path or dataframe must be provided')
    content = tabulate(df, headers='keys', tablefmt='pretty')
    if print_:
        print(content)
    return content
