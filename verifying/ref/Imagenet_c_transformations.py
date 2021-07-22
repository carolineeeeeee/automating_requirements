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

from src.constant import __root__

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB) # not needed for the image generation part
# Constants

# sample transformation parameter values with in the specified range, here chosen to be evenly distributed
# obtained through sampling images and checking the vd value in the requirements
intensity_shift_params = list(range(-120, 121))
# gaussian_noise_params = list(range(4, 49))
gamma_params = [x / 100 for x in list(range(90, 109))]
contrast_params = [x / 10 for x in list(range(1, 10))]
uniform_noise_params = [x / 10 for x in list(range(0, 7))]
lowpass_params = [x / 10 for x in list(range(0, 30))]
highpass_params = [x / 100 for x in list(range(0, 150))]
phase_noise_params = [x / 100 for x in list(range(0, 200))]


# transformations
def intensity_shift(img, degree, precision=0):
    return (img + np.around(degree, precision)).clip(min=0, max=255)


'''
def gaussian_noise(img, state, img_h, img_w, ch, precision=0): 
	# currently gaussian with mean 0
	if ch > 1:
		gauss = np.random.normal(0,state,(img_h, img_w, ch))
	else:
		gauss = np.random.normal(0,state,(img_h, img_w))
	noisy = img + np.around(gauss, precision)
	return noisy.clip(min=0, max=255) 
'''


def adjust_gamma(img, gamma, precision=0):
    return np.around(exposure.adjust_gamma(img, gamma), precision)


def adjust_contrast(image, contrast_level):
    """
    Taken from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/code/image_manipulation.py
    """

    assert (contrast_level >= 0.0), "contrast_level too low."
    assert (contrast_level <= 1.0), "contrast_level too high."

    return (1 - contrast_level) / 2.0 + image.dot(contrast_level)


def apply_uniform_noise(image, low, high, rng=None):
    """
    Taken from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/code/image_manipulation.py
    """
    nrow = image.shape[0]
    ncol = image.shape[1]
    nch = image.shape[2]

    image = image / 255

    image = image + get_uniform_noise(low, high, nrow, ncol, nch, rng)  # clip values

    image = np.where(image < 0, 0, image)
    image = np.where(image > 1, 1, image)

    assert is_in_bounds(image, 0, 1), "values <0 or >1 occurred"

    image = image * 255
    return image


def get_uniform_noise(low, high, nrow, ncol, nch, rng=None):
    """
    Taken from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/code/image_manipulation.py
    """
    if rng is None:
        return np.random.uniform(low=low, high=high,
                                 size=(nrow, ncol, nch))
    else:
        return rng.uniform(low=low, high=high,
                           size=(nrow, ncol, nch))


def is_in_bounds(mat, low, high):
    """
    Taken from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/code/image_manipulation.py
    """
    return np.all(np.logical_and(mat >= 0, mat <= 1))


def low_pass_filter(image, std):
    """
    Taken from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/code/image_manipulation.py
    """
    # set this to mean pixel value over all images
    bg_grey = 0.4423
    image = image / 255
    # covert image to greyscale and define variable prepare new image
    # image = rgb2grey(image)
    new_image = np.zeros(image.shape, image.dtype)

    # aplly Gaussian low-pass filter
    new_image = gaussian_filter(image, std, mode='constant', cval=bg_grey)

    # crop too small and too large values
    # new_image[new_image < 0] = 0
    # new_image[new_image > 1] = 1

    # return stacked (RGB) grey image
    # return np.dstack((new_image,new_image,new_image))
    new_image = new_image * 255
    return new_image.clip(min=0, max=255)


def high_pass_filter(image, std):
    """
        Taken from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/code/image_manipulation.py
    """

    # set this to mean pixel value over all images
    bg_grey = 0.4423
    image = image / 255

    # convert image to greyscale and define variable prepare new image
    # image = rgb2grey(image)
    new_image = np.zeros(image.shape, image.dtype)

    # aplly the gaussian filter and subtract from the original image
    gauss_filter = gaussian_filter(image, std, mode='constant', cval=bg_grey)
    new_image = image - gauss_filter

    # add mean of old image to retain image statistics
    mean_diff = bg_grey - np.mean(new_image, axis=(0, 1))
    new_image = new_image + mean_diff

    # crop too small and too large values
    # new_image[new_image < 0] = 0
    # new_image[new_image > 1] = 1

    # return stacked (RGB) grey image
    # return np.dstack((new_image,new_image,new_image))
    new_image = new_image * 255
    return new_image.clip(min=0, max=255)


def scramble_phases(image, width):
    """
    Taken from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/code/image_manipulation.py
    """
    image = image / 255
    # create array with random phase shifts from the interval [-width,width]
    length = (image.shape[0] - 1) * (image.shape[1] - 1)
    phase_shifts = np.random.random(length // 2) - 0.5
    phase_shifts = phase_shifts * 2 * width / 180 * np.pi

    # convert to graysclae
    channel = rgb2grey(image)
    # channel = image
    # print(channel.shape)

    # Fourier Forward Tranform and shift to centre
    f = fp.fft2(channel)  # rfft for real values
    f = fp.fftshift(f)

    # get amplitudes and phases
    f_amp = np.abs(f)
    f_phase = np.angle(f)

    # transformations of phases
    # just change the symmetric parts of FFT outcome, which is
    # [1:,1:] for the case of even image sizes
    fnew_phase = f_phase
    # print(f_phase.shape)
    # print(phase_shifts.shape)
    fnew_phase[1:, 1:] = shift_phases(f_phase[1:, 1:], phase_shifts)

    # recalculating FFT complex representation from new phases and amplitudes
    fnew = f_amp * np.exp(1j * fnew_phase)

    # reverse shift to centre and perform Fourier Backwards Transformation
    fnew = fp.ifftshift(fnew)
    new_channel = fp.ifft2(fnew)

    # make sure that there are no imaginary parts after transformation
    new_channel = new_channel.real

    # clip too small and too large values
    new_channel[new_channel > 1] = 1
    new_channel[new_channel < 0] = 0

    new_channel = new_channel * 255
    # return stacked (RGB) grey image
    return np.dstack((new_channel, new_channel, new_channel))


def shift_phases(f_phase, phase_shifts):
    """
    Taken from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/code/image_manipulation.py
    """

    # flatten array for easier transformation
    f_shape = f_phase.shape
    flat_phase = f_phase.flatten()
    length = flat_phase.shape[0]

    # apply phase shifts symmetrically to complex conjugate frequency pairs
    # do not change c-component
    # print(phase_shifts.shape)
    # print(flat_phase[:length//2].shape)
    if length % 2 == 1:
        flat_phase[:length // 2] += phase_shifts
        flat_phase[length // 2 + 1:] -= phase_shifts
    else:
        flat_phase[:length // 2] += phase_shifts
        flat_phase[length // 2:] -= phase_shifts

    # reshape into output format
    f_phase = flat_phase.reshape(f_shape)

    return f_phase


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


# def shot_noise(x, severity=1):
def shot_noise(x, i):
    #	c = [60, 25, 12, 5, 3][severity - 1]
    c = np.linspace(1, 1000, TRANSFORMATION_LEVEL)

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c[i]) / c[i], 0, 1) * 255, c[i]


# def impulse_noise(x, severity=1):
def impulse_noise(x, i):
    #	c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    c = np.linspace(0.0, 0.8, TRANSFORMATION_LEVEL)

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c[i])
    return np.clip(x, 0, 1) * 255, c[i]


# def speckle_noise(x, severity=1):
def speckle_noise(x, i):
    #	c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    c = [0.15, 0.2, 0.35, 0.45, 0.6]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c[i]), 0, 1) * 255, c[i]


# not inlcuded
def fgsm(x, source_net, severity=1):
    c = [8, 16, 32, 64, 128][severity - 1]

    x = V(x, requires_grad=True)
    logits = source_net(x)
    source_net.zero_grad()
    loss = F.cross_entropy(logits, V(logits.data.max(1)[1].squeeze_()), size_average=False)
    loss.backward()

    return standardize(torch.clamp(unstandardize(x.data) + c / 255. * unstandardize(torch.sign(x.grad.data)), 0, 1))


# def gaussian_blur(x, severity=1):
def gaussian_blur(x, i):
    #	c = [1, 2, 3, 4, 6][severity - 1]
    c = [1, 2, 3, 4, 6]

    x = gaussian(np.array(x) / 255., sigma=c[i], multichannel=True)
    return np.clip(x, 0, 1) * 255, c[i]


# def glass_blur(x, severity=1):
def glass_blur(x, i):
    # sigma, max_delta, iterations
    #	c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    sigma = np.linspace(0, 5, TRANSFORMATION_LEVEL)
    max_delta = np.random.randint(1, 4, TRANSFORMATION_LEVEL)
    iterations = 2
    c = np.stack([sigma, max_delta], 1)

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[i], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(iterations):
        for h in range(224 - max_delta[i], max_delta[i], -1):
            for w in range(224 - max_delta[i], max_delta[i], -1):
                dx, dy = np.random.randint(-max_delta[i], max_delta[i], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=sigma[i], multichannel=True), 0, 1) * 255, c[i]


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


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# def motion_blur(x, severity=1):
def motion_blur(x, i):
    #	c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

    radius = np.linspace(1, 20, TRANSFORMATION_LEVEL)
    sigma = np.linspace(1, 20, TRANSFORMATION_LEVEL)
    c = np.stack([radius, sigma], 1)

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[i][0], sigma=c[i][1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    if x.shape != (224, 224):
        return np.clip(x[..., [2, 1, 0]], 0, 255), c[i]  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255), c[i]


def clipped_zoom(img, zoom_factor):
    h, w, c = img.shape
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))
    top = (h - ch) // 2
    new_img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (new_img.shape[0] - h) // 2
    result_image = new_img[trim_top:trim_top + h, trim_top:trim_top + h]
    # pad with zeros for addition later
    img[trim_top:trim_top + h, trim_top:trim_top + h] = result_image
    return img


def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


# def fog(x, severity=1):
def fog(x, i):

    # c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]

    scale = np.linspace(1, 5, TRANSFORMATION_LEVEL)
    wibbledecay = np.linspace(5, 1, TRANSFORMATION_LEVEL)
    c = np.stack([scale, wibbledecay], 1)

    x = np.array(x) / 255.
    max_val = x.max()
    # print(x.shape)
    h, w, ch = x.shape
    plasma_size = w.bit_length()
    x += c[i][0] * plasma_fractal(mapsize=2 ** plasma_size, wibbledecay=c[i][1])[:h, :w][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[i][0]), 0, 1) * 255, c[i]


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
    frost = Image.open(os.path.join(__root__, 'frost-images', filename))

    # print(frost)
    x = np.asarray(x)
    h, w, ch = x.shape
    frost = np.asarray(frost.resize((w, h)))
    # randomly crop and convert to rgb
    frost = frost[..., [2, 1, 0]]
    # x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
    # frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]

    return np.clip(c[i][0] * x + c[i][1] * frost, 0, 255), c[i]


def snow(x, i):
    # c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
    #	 (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
    #	 (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
    #	 (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
    #	 (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    location = np.linspace(0, 1, TRANSFORMATION_LEVEL)
    scale = np.linspace(0, 1, TRANSFORMATION_LEVEL)
    zoom = np.linspace(0, 4, TRANSFORMATION_LEVEL)
    snow_layer_threshold = np.linspace(0, 1, TRANSFORMATION_LEVEL)
    blur_radius = np.linspace(1, 20, TRANSFORMATION_LEVEL)
    blur_sigma = np.linspace(1, 20, TRANSFORMATION_LEVEL)
    c = np.stack([location, scale, snow_layer_threshold, blur_radius, blur_sigma], 1)

    x = np.array(x, dtype=np.float32) / 255.
    h, w, ch = x.shape
    snow_layer = np.random.normal(size=(w, w), loc=c[i][0], scale=c[i][1])  # [:2] for monochrome
    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[i][2])

    snow_layer[snow_layer < c[i][3]] = 0

    snow_layer = Image.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[i][4], sigma=c[i][5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.frombuffer(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]
    x = c[i][6] * x + (1 - c[i][6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(h, w, 1) * 1.5 + 0.5)
    snow_layer = snow_layer[w // 2 - h // 2:w // 2 + (h - h // 2), 0:w]
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


# not used
def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


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


# def saturate(x, severity=1):
def saturate(x, i):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[i][0] + c[i][1], 0, 1)
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


# def pixelate(x, severity=1):
def pixelate(x, c=0.6):
    # c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    x = x.resize((int(224 * c), int(224 * c)), Image.BOX)
    x = x.resize((224, 224), Image.BOX)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
# this is geometric, let's not use it
def elastic_transform(image, severity=1):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),  # 244 should have been 224, but ultimately nothing is incorrect
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255
