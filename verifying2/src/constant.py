import os
import pathlib2
from torchvision import transforms

ROOT = pathlib2.Path(__file__).parent.parent

MIN_IQA_RANGE = 0.5
IQA = 'vif/vifvec_release'
IQA_PATH = os.path.join(ROOT, 'utils', 'image-quality-tools', 'metrix_mux', 'metrix', IQA)
matlabPyrToolsPath = os.path.join(IQA_PATH, "matlabPyrTools")

ROBUSTBENCH_CIFAR10_MODEL_NAMES = [
    "Hendrycks2020AugMix_WRN", "Gowal2020Uncovering_L2_70_16_extra", "Hendrycks2020AugMix_ResNeXt",
    "Gowal2020Uncovering_Linf_70_16_extra", "Kireev2021Effectiveness_Gauss50percent",
    "Kireev2021Effectiveness_RLATAugMixNoJSD", "Kireev2021Effectiveness_AugMixNoJSD", "Gowal2020Uncovering_Linf_70_16",
    "Gowal2020Uncovering_L2_70_16", "Calian2021Defending", "Kireev2021Effectiveness_RLAT", "Standard"]
# dataset
CIFAR10 = 'cifar10'
CIFAR10C = 'cifar10c'
# transformations
CONTRAST_G = "contrast_G"
UNIFORM_NOISE = "uniform_noise"
LOWPASS = "lowpass"
HIGHPASS = "highpass"
PHASE_NOISE = "phase_noise"
GAUSSIAN_NOISE = "gaussian_noise"
SHOT_NOISE = "shot_noise"
IMPULSE_NOISE = "impulse_noise"
DEFOCUS_BLUR = "defocus_blur"
GLASS_BLUR = "glass_blur"
MOTION_BLUR = "motion_blur"
SNOW = "snow"
FROST = "frost"
FOG = "fog"
BRIGHTNESS = "brightness"
CONTRAST = "contrast"
JPEG_COMPRESSION = "jpeg_compression"
GENERALIZED = 'generalized'
TRANSFORMATIONS = [CONTRAST_G, UNIFORM_NOISE, LOWPASS, HIGHPASS, PHASE_NOISE, GAUSSIAN_NOISE, SHOT_NOISE, IMPULSE_NOISE,
                   DEFOCUS_BLUR, GLASS_BLUR, MOTION_BLUR, SNOW, FROST, FOG, BRIGHTNESS, CONTRAST, JPEG_COMPRESSION]
CIFAR10_CLASSES = ['cat', 'ship', 'plane', 'frog', 'car', 'truck', 'dog', 'horse', 'deer', 'bird']
CIFAR10_C_CORRUPTION = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur", "gaussian_noise",
    "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow",
    "spatter", "speckle_noise", "zoom_blur"]
CIFAR10_INDEX_TO_CLASS = {i: category for i, category in enumerate(CIFAR10_CLASSES)}
CIFAR10_CLASS_to_INDEX = {category: i for i, category in enumerate(CIFAR10_CLASSES)}

THRESHOLD_MAP = {}
for tran in [
        CONTRAST_G, CONTRAST, UNIFORM_NOISE, LOWPASS, HIGHPASS, PHASE_NOISE, DEFOCUS_BLUR, MOTION_BLUR, GLASS_BLUR,
        GENERALIZED]:
    THRESHOLD_MAP[tran] = {}
THRESHOLD_MAP[CONTRAST_G]['abs'] = 0.89
THRESHOLD_MAP[CONTRAST]['abs'] = 0.89
THRESHOLD_MAP[UNIFORM_NOISE]['abs'] = 0.85
THRESHOLD_MAP[LOWPASS]['abs'] = 0.92
THRESHOLD_MAP[HIGHPASS]['abs'] = 0.86
THRESHOLD_MAP[PHASE_NOISE]['abs'] = 0.88
THRESHOLD_MAP[DEFOCUS_BLUR]['abs'] = 0.9
THRESHOLD_MAP[MOTION_BLUR]['abs'] = 0.9
THRESHOLD_MAP[GLASS_BLUR]['abs'] = 0.8
THRESHOLD_MAP[GENERALIZED]['abs'] = 0.85

THRESHOLD_MAP[CONTRAST_G]['rel'] = 0.99
THRESHOLD_MAP[CONTRAST]['rel'] = 0.99
THRESHOLD_MAP[UNIFORM_NOISE]['rel'] = 0.83
THRESHOLD_MAP[LOWPASS]['rel'] = 0.91
THRESHOLD_MAP[HIGHPASS]['rel'] = 0.98
THRESHOLD_MAP[PHASE_NOISE]['rel'] = 0.86
THRESHOLD_MAP[DEFOCUS_BLUR]['rel'] = 1
THRESHOLD_MAP[MOTION_BLUR]['rel'] = 0.85
THRESHOLD_MAP[GLASS_BLUR]['rel'] = 0.8
THRESHOLD_MAP[GENERALIZED]['rel'] = 0.83

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
MODELS = [ALEXNET, RESNET50, RESNEXT50, VGG_16, GOOGLENET]

IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

IMAGENET_DEFAULT_TRANSFORMATION = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    IMAGENET_NORMALIZE,
])

CIFAR10_DEFAULT_TRANSFORMATION = transforms.Compose(
    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
