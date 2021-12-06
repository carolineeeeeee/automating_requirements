# Transformations obtained from https://github.com/hendrycks/robustness with slight modification
# to adapt to images with different sizes and random sampling

import cv2
import logging
import numpy as np
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
from io import BytesIO
from PIL import Image
from wand.image import Image as WandImage
import os
import sys
import random
import pathlib2
import wand.color as WandColor
from scipy import fftpack as fp
from shutil import copy as copy_file
from wand.api import library as wandlibrary
from skimage.color import rgb2gray, rgb2grey
from scipy.ndimage.filters import gaussian_filter

from .constant import ROOT, DATA_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# /////////////// Distortions ///////////////
TRANSFORMATION_LEVEL = 1000


def save_array(dest, arr):
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(dest)


def gaussian_noise(x, i):
    # c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    c = np.linspace(0.08, 0.9, TRANSFORMATION_LEVEL)

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c[i]), 0, 1) * 255, c[i]

def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

     # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

# def defocus_blur(x, severity=1):
def defocus_blur(x, i):
    #	c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    radius = np.linspace(1, 10, TRANSFORMATION_LEVEL)
    alias_blur = np.linspace(0, 1, TRANSFORMATION_LEVEL)
    c = np.stack([radius, alias_blur], 1)

    x = np.array(x) / 255.
    kernel = disk(radius=c[i][0], alias_blur=c[i][1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255, c[i]


# def frost(x, severity=1):
def frost(x, i):
    # c = [(1, 0.4),
    #	 (0.8, 0.6),
    #	 (0.7, 0.7),
    #	 (0.65, 0.7),
    #	 (0.6, 0.75)]

    scale = np.linspace(0.01, 1, TRANSFORMATION_LEVEL)
    constant = np.linspace(0.01, 1, TRANSFORMATION_LEVEL)
    c = np.stack([scale, constant], 1)

    idx = np.random.randint(5)
    filename = ['frost1.png', 'frost2.png', 'frost3.png', 'frost4.jpeg', 'frost5.jpeg', 'frost6.jpeg'][idx]
    frost = Image.open(os.path.join(DATA_DIR,  'frost-images', filename))

    # print(frost)
    x = np.asarray(x)
    h, w, ch = x.shape
    frost = np.asarray(frost.resize((w, h)))
    # randomly crop and convert to rgb
    frost = frost[..., [2, 1, 0]]
    # x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
    # frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]

    return np.clip(c[i][0] * x + c[i][1] * frost, 0, 255), c[i]


def contrast(x, i):
    # def contrast(x, severity=1):
    # c = [0.4, .3, .2, .1, .05]
    c = np.linspace(0.01, 0.9, TRANSFORMATION_LEVEL)

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c[i] + means, 0, 1) * 255, c[i]


# def brightness(x, severity=1):
def brightness(x, i):
    # c = [.1, .2, .3, .4, .5]
    c = np.linspace(0, 1, TRANSFORMATION_LEVEL)

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c[i], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255, c[i]



# def jpeg_compression(x, severity=1):
def jpeg_compression(x, i) -> Image:
    # c = [25, 18, 15, 10, 7][severity - 1]
    # c = [25, 18, 15, 10, 7]
    c = list(range(1, (TRANSFORMATION_LEVEL + 1)))

    output = BytesIO()
    x.save(output, 'JPEG', quality=c[i])
    x = Image.open(output)

    return x, c[i]

#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.ColorJitter
def Color_jitter (x, i):
    #b = np.linspace(0, 1, 100)
    color_jitter = A.ReplayCompose([ColorJitter(brightness=(0.5,1), contrast=(0.5,1), saturation=(0.5,1), hue=(0.5,1), always_apply=True)])
    transformed_img = color_jitter(image=x)
    # to get the parameters that are actually used
    arguments = str(transformed_img['replay']['transforms'][0]['params']['contrast']) + ", " + str(transformed_img['replay']['transforms'][0]['params']['saturation']) + ", " + str(transformed_img['replay']['transforms'][0]['params']['hue'])
    
    return transformed_img['image'], arguments
    
#https://albumentations.ai/docs/api_reference/augmentations/transforms/#albumentations.augmentations.transforms.RGBShift
def RGB (x,i):
    RGB_Shift = A.ReplayCompose([RGBShift(r_shift_limit=[150, 400], g_shift_limit=[150, 400], b_shift_limit=[150, 400], always_apply=True)])
    transformed_img = RGB_Shift(image=x)
    # to get the parameters that are actually used
    arguments = str(transformed_img['replay']['transforms'][0]['params']['r_shift']) + ", " +  str(transformed_img['replay']['transforms'][0]['params']['g_shift']) +", " +  str(transformed_img['replay']['transforms'][0]['params']['b_shift'])
    
    return transformed_img['image'], arguments
