import pathlib2
from torchvision import transforms
from collections import defaultdict
__root__ = pathlib2.Path(__file__).absolute().parent.parent
ROOT_PATH = __root__

MIN_IQA_RANGE = 0.5

OBJECT_CLASSES = ["airplane", "bicycle", "boat", "car", "chair", "dog", "keyboard", "oven", "bear", "bird",
                  "bottle", "cat", "clock", "elephant", "knife", "truck"]
CIFAR10_CLASSES = ['cat', 'ship', 'plane', 'frog', 'car', 'truck', 'dog', 'horse', 'deer', 'bird']
#ROBUSTBENCH_CIFAR10_MODEL_NAMES = [
#    "Carmon2019Unlabeled", "Chen2020Adversarial", "Chen2020Efficient", "Cui2020Learnable_34_10",
#    "Cui2020Learnable_34_20", "Ding2020MMA", "Engstrom2019Robustness", "Gowal2020Uncovering_28_10_extra",
#    "Gowal2020Uncovering_34_20", "Sehwag2020Hydra", "Sehwag2021Proxy_R18", "Standard", "Wu2020Adversarial",
#    "Wu2020Adversarial_extra", "Zhang2019You", "Zhang2020Attacks"]
ROBUSTBENCH_CIFAR10_MODEL_NAMES = ["Hendrycks2020AugMix_ResNeXt", "Hendrycks2020AugMix_WRN", 
    "Kireev2021Effectiveness_RLATAugMixNoJSD", "Kireev2021Effectiveness_AugMixNoJSD","Kireev2021Effectiveness_Gauss50percent",
    "Kireev2021Effectiveness_RLAT", "Standard"]
    
CIFAR10_INDEX_TO_CLASS = {i: category for i, category in enumerate(CIFAR10_CLASSES)}
CIFAR10_CLASS_to_INDEX = {category: i for i, category in enumerate(CIFAR10_CLASSES)}

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
CIFAR10_C_CORRUPTION = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur", "gaussian_noise",
    "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow",
    "spatter", "speckle_noise", "zoom_blur"]


THRESHOLD_MAP = {}
for tran in [CONTRAST_G, CONTRAST, UNIFORM_NOISE, LOWPASS, HIGHPASS, PHASE_NOISE, DEFOCUS_BLUR, MOTION_BLUR, GLASS_BLUR, GENERALIZED]:
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

# models
ALEXNET = 'alexnet'
DARKNET19 = 'darknet19'
DARKNET53_448 = 'darknet53_448'
RESNET50 = 'resnet50'
RESNEXT50 = 'resnext50'
VGG_16 = 'vgg-16'
GOOGLENET = 'googlenet'
MODELS = [ALEXNET, RESNET50, RESNEXT50, VGG_16, GOOGLENET]
# MODELS = ['Standard', 'Engstrom2019Robustness', 'Rice2020Overfitting', 'Carmon2019Unlabeled']
CIFAR10_MODELS = ['Carmon2019Unlabeled']

# transformation to beperformed on a image to make it compatible with pytorch framework
# images have to be the same dimension in order to be stacked together and passed to model to be evaluated in parallel
IMAGE_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
IMAGENET_DEFAULT_TRANSFORMATION = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    IMAGE_NORMALIZE,
])


CIFAR10_DEFAULT_TRANSFORMATION = transforms.Compose(
    [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
