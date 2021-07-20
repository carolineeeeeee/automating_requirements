from torchvision import transforms

OBJECT_CLASSES = ["airplane", "bicycle", "boat", "car", "chair", "dog", "keyboard", "oven", "bear", "bird",
                  "bottle", "cat", "clock", "elephant", "knife", "truck"]
CIFAR10_CLASSES = ['cat', 'ship', 'plane', 'frog', 'car', 'truck', 'dog', 'horse', 'deer', 'bird']
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
TRANSFORMATIONS = [CONTRAST_G, UNIFORM_NOISE, LOWPASS, HIGHPASS, PHASE_NOISE, GAUSSIAN_NOISE, SHOT_NOISE,
                   IMPULSE_NOISE, DEFOCUS_BLUR, SNOW, FROST, FOG, BRIGHTNESS, CONTRAST]

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
