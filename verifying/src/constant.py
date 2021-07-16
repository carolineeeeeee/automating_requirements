from torchvision import transforms

OBJECT_CLASSES = ["airplane", "bicycle", "boat", "car", "chair", "dog", "keyboard", "oven", "bear", "bird",
                  "bottle", "cat", "clock", "elephant", "knife", "truck"]
CIFAR10_CLASSES = ['cat', 'ship', 'plane', 'frog', 'car', 'truck', 'dog', 'horse', 'deer', 'bird']
# transformations
INTENSITY_SHIFT = "intensity_shift"
GAUSSIAN_NOISE = "gaussian_noise"
GAMMA = "gamma"
CONTRAST = "contrast"
UNIFORM_NOISE = "uniform_noise"
LOWPASS = "lowpass"
HIGHPASS = "highpass"
PHASE_NOISE = "phase_noise"
TRANSFORMATIONS = [INTENSITY_SHIFT, GAUSSIAN_NOISE, GAMMA, CONTRAST, UNIFORM_NOISE, LOWPASS, HIGHPASS, PHASE_NOISE]

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

# sample transformation parameter values with in the specified range, here chosen to be evenly distributed
# obtained through sampling images and checking the vd value in the requirements
INTENSITY_SHIFT_PARAMS_DOMAIN = (-120, 120)
GAUSSIAN_NOISE_PARAMS_DOMAIN = (4, 48)
GAMMA_PARAMS_DOMAIN = (0.9, 1.09)
CONTRAST_PARAMS_DOMAIN = (0, 1)
UNIFORM_NOISE_PARAMS_DOMAIN = (0, 0.7)
LOWPASS_PARAMS_DOMAIN = (0, 3)
HIGHPASS_PARAMS_DOMAIN = (0, 15)
PHASE_NOISE_PARAMS_DOMAIN = (0, 90)
TRANSFORMATION_PARAM_MAP_DOMAIN = {
    INTENSITY_SHIFT: INTENSITY_SHIFT_PARAMS_DOMAIN,
    GAUSSIAN_NOISE: GAUSSIAN_NOISE_PARAMS_DOMAIN,
    GAMMA: GAMMA_PARAMS_DOMAIN,
    CONTRAST: CONTRAST_PARAMS_DOMAIN,
    UNIFORM_NOISE: UNIFORM_NOISE_PARAMS_DOMAIN,
    LOWPASS: LOWPASS_PARAMS_DOMAIN,
    HIGHPASS: HIGHPASS_PARAMS_DOMAIN,
    PHASE_NOISE: PHASE_NOISE_PARAMS_DOMAIN
}
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

# static filenames
CLASS_TO_LABEL_ID_JSON = "class_to_label_id.json"
LABEL_ID_TO_LABELS_JSON = "label_id_to_labels.json"
INDEX_TO_LABELS_JSON = "index_to_labels.json"
INDEX_TO_LABEL_ID_JSON = "index_to_label_id.json"
IMAGE_TO_LABEL_ID_CSV = "image_to_label_id.csv"
INET_VAL_LIST_FILENAME = "inet.val.list"
SYNSET_WORDS_TXT = "synset_words.txt"
MSCOCO_TO_IMAGENET_CATEGORY_MAPPING_TXT = 'MSCOCO_to_ImageNet_category_mapping.txt'
