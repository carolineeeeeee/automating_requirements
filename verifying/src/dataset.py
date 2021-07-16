import os
import cv2
import torch
import pathlib2
import torchvision
import numpy as np
import pandas as pd
from typing import Union, Dict
from torchvision import transforms
from torch.utils.data import Dataset

from .constant import GAUSSIAN_NOISE, \
    INTENSITY_SHIFT, \
    GAMMA, \
    CONTRAST, \
    UNIFORM_NOISE, \
    LOWPASS, \
    HIGHPASS, \
    PHASE_NOISE
from src.transformation import gaussian_noise, intensity_shift, adjust_gamma, adjust_contrast, \
    apply_uniform_noise, low_pass_filter, high_pass_filter, scramble_phases

ToTensor = transforms.ToTensor()


def apply_transformation(img: np.ndarray, transformation_type: str, param: float) -> np.ndarray:
    """apply specified transformation to a given image with given transformation parameters

    :param img: image which transformation will be applied torch
    :type img: np.ndarray
    :param transformation_type: type of transformation to performance_df
    :type transformation_type: str
    :param param: transformation parameter, a value specifying how much transformation to perform
    :type param: float
    :raises ValueError: Wrong transformation type passed intensity_shift
    :return: transformed images
    :rtype: np.ndarray
    """
    if transformation_type == GAUSSIAN_NOISE:
        img2 = gaussian_noise(img, param)
    elif transformation_type == INTENSITY_SHIFT:
        img2 = intensity_shift(img, param)
    elif transformation_type == GAMMA:
        img2 = adjust_gamma(img, param)
    elif transformation_type == CONTRAST:
        img2 = adjust_contrast(img, param)
    elif transformation_type == UNIFORM_NOISE:
        img2 = apply_uniform_noise(img, 0, param)
    elif transformation_type == LOWPASS:
        img2 = low_pass_filter(img, param)
    elif transformation_type == HIGHPASS:
        img2 = high_pass_filter(img, param)
    elif transformation_type == PHASE_NOISE:
        img2 = scramble_phases(img, param)
    else:
        raise ValueError("Invalid transformation type")
    return img2


class ImagenetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: Union[str, pathlib2.Path], transform: transforms.Compose) -> None:
        """Initializer of ImagenetDataset Class

        :param df: a table/DataFrame containing info for images: image name, target, transformation type and parameter
        :type df: pd.DataFrame
        :param image_root: directory of images, used for constructing full absolute image path
        :type image_root: str
        :param transform: transformation to perform on images, not the same as the one in df, e.g. resizing, toTensor
        :type transform: transforms.Compose
        """
        self.image_root = os.path.abspath(image_root)
        self.df = df.reset_index()
        self.df['image_path'] = self.df['image_name'].apply(
            lambda image_name: os.path.join(self.image_root, f"{image_name}.JPEG"))
        self.image_data_list = self.df.to_dict('records')
        self.transform = transform

    def __len__(self) -> int:
        """Get size of dataset

        :return: number of images in dataset
        :rtype: int
        """
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        """Get a transformed image tensor ready for training/eval as well as the metadata of the image

        :param index: index to image
        :type index: int
        :return: image tensor ready for training/eval as well as image's meta data
        :rtype: Dict
        """
        data = self.image_data_list[index]
        original_image = cv2.imread(data['image_path'])
        assert original_image is not None
        image = np.asarray(original_image, dtype=np.float32)
        transformed_image = apply_transformation(
            image, data['transformation_type'],
            data['transformation_param'])

        # convert transformed image array from BGR into RGB
        transformed_image = cv2.cvtColor(cv2.normalize(transformed_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                                         cv2.COLOR_BGR2RGB)
        return {
            'original_image': self.transform(ToTensor(original_image)),
            'transformed_image': self.transform(ToTensor(transformed_image)),
            'label_index': torch.tensor(data['label_index']),
            'image_name': data['image_name'],
            'image_path': data['image_path'],
            'label_id': data['label_id'],
            'transformation_type': data['transformation_type'],
            'transformation_param': data['transformation_param']
        }
