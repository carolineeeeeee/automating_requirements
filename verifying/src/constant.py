import os
import pathlib2
from torchvision import transforms

ROOT = pathlib2.Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'
IMAGENET_DATA_DIR = DATA_DIR / 'imagenet'
# ImageNet info path
IMAGE_2_LABEL_PATH = IMAGENET_DATA_DIR / 'info' / 'image_to_label_id.csv'

# requirement types
ACCURACY_PRESERVATION = "accuracy_preservation"
PREDICTION_PRESERVATION = "prediction_preservation"

MIN_IQA_RANGE = 0.5
IQA = 'vif/vifvec_release'
IQA_PATH = os.path.join(str(ROOT), 'utils', 'image-quality-tools', 'metrix_mux', 'metrix', IQA)
matlabPyrToolsPath = os.path.join(IQA_PATH, "matlabPyrTools")

ROBUSTBENCH_CIFAR10_MODEL_NAMES = [
    "Hendrycks2020AugMix_WRN", "Gowal2020Uncovering_L2_70_16_extra", "Hendrycks2020AugMix_ResNeXt",
    "Gowal2020Uncovering_Linf_70_16_extra", "Kireev2021Effectiveness_Gauss50percent",
    "Kireev2021Effectiveness_RLATAugMixNoJSD", "Kireev2021Effectiveness_AugMixNoJSD", "Gowal2020Uncovering_Linf_70_16",
    "Gowal2020Uncovering_L2_70_16", "Calian2021Defending", "Kireev2021Effectiveness_RLAT", "Standard"]
# dataset
CIFAR10 = 'cifar10'
CIFAR10C = 'cifar10c'
IMAGENET = 'imagenet'
# transformations
GAUSSIAN_NOISE = "gaussian_noise"
DEFOCUS_BLUR = "defocus_blur"
FROST = "frost"
BRIGHTNESS = "brightness"
CONTRAST = "contrast"
JPEG_COMPRESSION = "jpeg_compression"
RGB = "RGB"
COLOR_JITTER = "color_jitter"
TRANSFORMATIONS = [GAUSSIAN_NOISE, FROST, BRIGHTNESS, CONTRAST, JPEG_COMPRESSION, RGB, COLOR_JITTER, DEFOCUS_BLUR]
CIFAR10_CLASSES = ['cat', 'ship', 'plane', 'frog', 'car', 'truck', 'dog', 'horse', 'deer', 'bird']
CIFAR10_C_CORRUPTION = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur", "gaussian_noise",
    "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow",
    "spatter", "speckle_noise", "zoom_blur"]
CIFAR10_INDEX_TO_CLASS = {i: category for i, category in enumerate(CIFAR10_CLASSES)}
CIFAR10_CLASS_to_INDEX = {category: i for i, category in enumerate(CIFAR10_CLASSES)}

THRESHOLD_MAP = {}
for tran in TRANSFORMATIONS:
    THRESHOLD_MAP[tran] = {}

THRESHOLD_MAP[RGB][ACCURACY_PRESERVATION] = 0.64
THRESHOLD_MAP[JPEG_COMPRESSION][ACCURACY_PRESERVATION] = 0.92
THRESHOLD_MAP[COLOR_JITTER][ACCURACY_PRESERVATION] = 0.7
THRESHOLD_MAP[DEFOCUS_BLUR][ACCURACY_PRESERVATION] = 0.97
THRESHOLD_MAP[GAUSSIAN_NOISE][ACCURACY_PRESERVATION] = 0.97
THRESHOLD_MAP[CONTRAST][ACCURACY_PRESERVATION] = 0.76
THRESHOLD_MAP[FROST][ACCURACY_PRESERVATION] = 0.83
THRESHOLD_MAP[BRIGHTNESS][ACCURACY_PRESERVATION] = 0.86

THRESHOLD_MAP[RGB][PREDICTION_PRESERVATION] = 0.73
THRESHOLD_MAP[JPEG_COMPRESSION][PREDICTION_PRESERVATION] = 0.9
THRESHOLD_MAP[COLOR_JITTER][PREDICTION_PRESERVATION] = 0.7
THRESHOLD_MAP[DEFOCUS_BLUR][PREDICTION_PRESERVATION] = 0.96
THRESHOLD_MAP[GAUSSIAN_NOISE][PREDICTION_PRESERVATION] = 0.94
THRESHOLD_MAP[CONTRAST][PREDICTION_PRESERVATION] = 0.84
THRESHOLD_MAP[FROST][PREDICTION_PRESERVATION] = 0.86
THRESHOLD_MAP[BRIGHTNESS][PREDICTION_PRESERVATION] = 0.86
TRANSFORMATION_LEVEL = 1000

intensity_shift_params = list(range(-120, 121))
# gaussian_noise_params = list(range(4, 49))
gamma_params = [x / 100 for x in list(range(90, 109))]
contrast_params = [x / 10 for x in list(range(1, 10))]
uniform_noise_params = [x / 10 for x in list(range(0, 7))]
lowpass_params = [x / 10 for x in list(range(0, 30))]
highpass_params = [x / 100 for x in list(range(0, 150))]
phase_noise_params = [x / 100 for x in list(range(0, 200))]

# ImageNet Models
ALEXNET = 'alexnet'
DARKNET19 = 'darknet19'
DARKNET53_448 = 'darknet53_448'
RESNET50 = 'resnet50'
RESNEXT50 = 'resnext50'
VGG_16 = 'vgg-16'
GOOGLENET = 'googlenet'
IMAGENET_MODELS = [ALEXNET, RESNET50, RESNEXT50, VGG_16, GOOGLENET]

IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

IMAGENET_DEFAULT_TRANSFORMATION = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    IMAGENET_NORMALIZE,
])

CIFAR10_DEFAULT_TRANSFORMATION = transforms.Compose(
    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
