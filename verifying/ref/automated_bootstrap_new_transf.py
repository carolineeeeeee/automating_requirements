import numpy as np
import cv2
import sys
import math
import os
from os import listdir
from os.path import isfile, join
import re
from shutil import copyfile
import pickle
from scipy.stats import norm
from scipy.integrate import quad
import scipy
import random
from skimage import exposure
from glob import glob
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import time
import resource
import getopt
import argparse

import matlab.engine
from Imagenet_c_transformations import *

__dir__ = os.path.dirname(os.path.abspath(__file__))

IMAGES = 'cifar-10'

IQA = 'vif/vifvec_release'
IQA_PATH = os.path.join(__dir__, 'image-quality-tools/metrix_mux/metrix/vif/vifvec_release')
matlabPyrToolsPath = os.path.join(__dir__, "image-quality-tools/metrix_mux/metrix/vif/vifvec_release/matlabPyrTools")


list_transformations_G = ['contrast_G', 'uniform_noise', 'lowpass', 'highpass', 'phase_noise']
t_params = {}
t_params['contrast_G'] = {}
t_params['uniform_noise'] = {}
t_params['lowpass'] = {}
t_params['highpass'] = {}
t_params['phase_noise'] = {}
t_params['generalized'] = {}

t_params['contrast_G']['abs'] = 0.89
t_params['uniform_noise']['abs'] = 0.85
t_params['lowpass']['abs'] = 0.92
t_params['highpass']['abs'] = 0.86
t_params['phase_noise']['abs'] = 0.88
t_params['generalized']['abs'] = 0.85

t_params['contrast_G']['rel'] = 0.99
t_params['uniform_noise']['rel'] = 0.83
t_params['lowpass']['rel'] = 0.91
t_params['highpass']['rel'] = 0.98
t_params['phase_noise']['rel'] = 0.86
t_params['generalized']['rel'] = 0.83

a_params = {}
a_params['contrast_G'] = 1.0
a_params['uniform_noise'] = 1.0
a_params['lowpass'] = 1.0
a_params['highpass'] = 1.0
a_params['phase_noise'] = 1.0
a_params['generalized'] = 1.0

list_transformations = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow',
    'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
'''
t_params = {}
t_params['defocus_blur'] = {}
t_params['Motion_Blur'] = {}
t_params['glass_blur'] = {}
t_params['motion_blur_v2'] = {}
t_params['Gaussian_blur'] = {}
t_params['generalized'] = {}

t_params['defocus_blur']['abs'] = 0.9
t_params['Motion_Blur']['abs'] = 0.9
t_params['glass_blur']['abs'] = 0.8
t_params['motion_blur_v2']['abs'] = 0.9
t_params['Gaussian_blur']['abs'] = 0.85
t_params['generalized']['abs'] = 0.81

t_params['defocus_blur']['rel'] = 1
t_params['Motion_Blur']['rel'] = 0.75
t_params['glass_blur']['rel'] = 0.8
t_params['motion_blur_v2']['rel'] = 0.85
t_params['Gaussian_blur']['rel'] = 0.8
t_params['generalized']['rel'] = 0.72

a_params = {}
a_params['defocus_blur'] = 1
a_params['Motion_Blur'] = 1
a_params['glass_blur'] = 0.84
a_params['motion_blur_v2'] = 0.92
a_params['Gaussian_blur'] = 0.91
a_params['generalized'] = 0.85
'''

# first function: sample from all the classes (/Users/caroline/Desktop/REforML/HVS/experiment_code/orig_images)
# 100 batches of 50 images


def filter_image_classes(JPEG_files):
    mapping_file = open("MSCOCO_to_ImageNet_category_mapping.txt", "r").readlines()
    labels = re.findall('(n\d+)', ' '.join(mapping_file))
    #JPEG_files = open("inet.val.list", "r").readlines()
    # print(len(JPEG_files))

    def find_label(filename):
        label = re.findall('\.(n\d+)\.', filename)[0]
        if label in labels:
            return True
        else:
            return False
    JPEG_files = [f for f in JPEG_files if find_label(f) == True]
    # print(len(JPEG_files))
    # print(JPEG_files[0])

    #list_to_filter = []
    # print(labels)
    # while True:
    #	line = mapping_file.readline()
    #	if not line:
    #		break

    return JPEG_files


def gen_bootstrapping(num_batch, orig_path, gen_path, t, save_name, batch_size=50, transformation='gaussian_noise'):
    if not os.path.exists(gen_path):
        os.makedirs(gen_path)
    if not os.path.exists(gen_path + save_name):
        os.makedirs(gen_path+save_name)
    gen_path = gen_path + save_name + '/'
    print(gen_path)

    # JPEG_files = [f for f in os.listdir(orig_path) if isfile(join(orig_path, f)) and f.endswith('JPEG')] #here it should be full path
    #list.sort(JPEG_files, key=find_num)
    JPEG_files = [y for x in os.walk(orig_path) for y in glob(os.path.join(x[0], '*.JPEG'))]
    #JPEG_files = filter_image_classes(JPEG_files)
    # print(len(JPEG_files))
    # exit()
    # filter out other classes
    eng = matlab.engine.start_matlab()
    eng.addpath(IQA_PATH, nargout=0)
    #eng.addpath(SSIM_PATH, nargout=0)
    eng.addpath(matlabPyrToolsPath, nargout=0)
    eng.addpath(matlabPyrToolsPath + '/MEX', nargout=0)

    set_orig = []
    set_transformed = []
    for i in range(num_batch):
        print("batch " + str(i))
        batch = random.sample(JPEG_files, batch_size)
        set_orig += batch
        if not os.path.exists(gen_path + "batch_" + str(i) + '/'):
            os.makedirs(gen_path + "batch_" + str(i) + '/')
        store_path = gen_path + "batch_" + str(i) + '/'
        j = 0
        for f in batch:
            #image_name = f[:-5]
            j += 1
            print(f)
            if IMAGES == "cifar-10":
                image_name = re.findall('(\d+)', f)[0]
            else:
                image_name = re.findall('(ILSVRC2012_val_\d+)', f)[0]
            print(image_name)
            if transformation == 'contrast_G':
                img = np.asarray(cv2.imread(f), dtype=np.float32)
                img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                while True:
                    param = random.choice(contrast_params)
                    img2 = adjust_contrast(img, param)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                        print(IQA_score)
                    except:
                        print("failed")
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        cv2.imwrite(store_path+'contrast_' + str(j) + "_" + image_name + '.JPEG', img2)
                        set_transformed.append(store_path + 'contrast_' + str(j) + "_" + image_name + '.JPEG')
                        break

            if transformation == 'uniform_noise':
                img = np.asarray(cv2.imread(f), dtype=np.float32)
                img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                while True:
                    param = random.choice(uniform_noise_params)
                    img2 = apply_uniform_noise(img, 0, param)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        cv2.imwrite(store_path+'uniform_noise_' + str(j) + "_" + image_name + '.JPEG', img2)
                        set_transformed.append(store_path + 'uniform_noise_' + str(j) + "_" + image_name + '.JPEG')
                        break

            if transformation == 'lowpass':
                img = np.asarray(cv2.imread(f), dtype=np.float32)
                img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                while True:
                    param = random.choice(lowpass_params)
                    img2 = low_pass_filter(img, param)
                    img2 = apply_uniform_noise(img, 0, param)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        cv2.imwrite(store_path+'lowpass_' + str(j) + "_" + image_name + '.JPEG', img2)
                        set_transformed.append(store_path + 'lowpass_' + str(j) + "_" + image_name + '.JPEG')
                        break

            if transformation == 'highpass':
                img = np.asarray(cv2.imread(f), dtype=np.float32)
                img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                while True:
                    param = random.choice(highpass_params)
                    img2 = high_pass_filter(img, param)
                    img2 = apply_uniform_noise(img, 0, param)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                        print(IQA_score)
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        cv2.imwrite(store_path+'highpass_' + str(j) + "_" + image_name + '.JPEG', img2)
                        set_transformed.append(store_path + 'highpass_' + str(j) + "_" + image_name + '.JPEG')
                        break

            if transformation == 'phase_noise':
                img = np.asarray(cv2.imread(f), dtype=np.float32)
                img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                while True:
                    param = random.choice(phase_noise_params)
                    img2 = scramble_phases(img, param)
                    img2 = apply_uniform_noise(img, 0, param)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                        print(IQA_score)
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        cv2.imwrite(store_path+'phase_noise_' + str(j) + "_" + image_name + '.JPEG', img2)
                        set_transformed.append(store_path + 'phase_noise_' + str(j) + "_" + image_name + '.JPEG')
                        break

            if transformation == 'gaussian_noise':
                img = Image.open(f)
                img_g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                while True:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, param = gaussian_noise(img, param_index)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                        print("debug")
                        print(IQA_score)
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        img2 = Image.fromarray(img2.astype(np.uint8))
                        img2.save(store_path+'gaussian_noise_' + str(j) + "_" + image_name + '.JPEG')
                        set_transformed.append(store_path + 'gaussian_noise_' + str(j) + "_" + image_name + '.JPEG')
                        #cv2.imwrite(store_path+'_motion_blur_v2_' + str(i) + "_" + image_name + '.png', img2)
                        break

            if transformation == 'shot_noise':
                img = Image.open(f)
                img_g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                while True:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, param = shot_noise(img, param_index)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        img2 = Image.fromarray(img2.astype(np.uint8))
                        img2.save(store_path+'shot_noise_' + str(j) + "_" + image_name + '.JPEG')
                        set_transformed.append(store_path + 'shot_noise_' + str(j) + "_" + image_name + '.JPEG')
                        #cv2.imwrite(store_path+'_motion_blur_v2_' + str(i) + "_" + image_name + '.png', img2)
                        break

            if transformation == 'impulse_noise':
                img = Image.open(f)
                img_g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                while True:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, param = impulse_noise(img, param_index)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        img2 = Image.fromarray(img2.astype(np.uint8))
                        img2.save(store_path+'impulse_noise_' + str(j) + "_" + image_name + '.JPEG')
                        set_transformed.append(store_path + 'impulse_noise_' + str(j) + "_" + image_name + '.JPEG')
                        #cv2.imwrite(store_path+'_motion_blur_v2_' + str(i) + "_" + image_name + '.png', img2)
                        break

            if transformation == 'defocus_blur':
                img = np.asarray(cv2.imread(f), dtype=np.float32)
                img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                while True:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, param = defocus_blur(img, param_index)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        cv2.imwrite(store_path+'defocus_blur_' + str(j) + "_" + image_name + '.JPEG', img2)
                        set_transformed.append(store_path + 'defocus_blur_' + str(j) + "_" + image_name + '.JPEG')
                        break

            if transformation == 'glass_blur':
                img = np.asarray(cv2.imread(f), dtype=np.float32)
                img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                while True:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, param = glass_blur(img, param_index)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        cv2.imwrite(store_path+'glass_blur_' + str(j) + "_" + image_name + '.JPEG', img2)
                        set_transformed.append(store_path + 'glass_blur_' + str(j) + "_" + image_name + '.JPEG')
                        break

            if transformation == 'motion_blur':
                img = Image.open(f)
                img_g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                while True:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2, param = motion_blur(img, param_index)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        img2 = Image.fromarray(img2.astype(np.uint8))
                        img2.save(store_path+'motion_blur_' + str(j) + "_" + image_name + '.JPEG')
                        set_transformed.append(store_path + 'motion_blur_' + str(j) + "_" + image_name + '.JPEG')
                        #cv2.imwrite(store_path+'_motion_blur_v2_' + str(i) + "_" + image_name + '.png', img2)
                        break

            if transformation == 'snow':
                img = Image.open(f)
                img_g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                while True:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2 = snow(img, param_index)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        img2 = Image.fromarray(img2.astype(np.uint8))
                        img2.save(store_path+'snow_' + str(j) + "_" + image_name + '.JPEG')
                        set_transformed.append(store_path + 'snow_' + str(j) + "_" + image_name + '.JPEG')
                        #cv2.imwrite(store_path+'_motion_blur_v2_' + str(i) + "_" + image_name + '.png', img2)
                        break

            if transformation == 'frost':
                img = Image.open(f)
                img_g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                while True:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2 = frost(img, param_index)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        img2 = Image.fromarray(img2.astype(np.uint8))
                        img2.save(store_path+'frost_' + str(j) + "_" + image_name + '.JPEG')
                        set_transformed.append(store_path + 'frost_' + str(j) + "_" + image_name + '.JPEG')
                        #cv2.imwrite(store_path+'_motion_blur_v2_' + str(i) + "_" + image_name + '.png', img2)
                        break

            if transformation == 'fog':
                img = Image.open(f)
                img_g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                while True:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2 = fog(img, param_index)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        img2 = Image.fromarray(img2.astype(np.uint8))
                        img2.save(store_path+'fog_' + str(j) + "_" + image_name + '.JPEG')
                        set_transformed.append(store_path + 'fog_' + str(j) + "_" + image_name + '.JPEG')
                        #cv2.imwrite(store_path+'_motion_blur_v2_' + str(i) + "_" + image_name + '.png', img2)
                        break

            if transformation == 'brightness':
                img = Image.open(f)
                img_g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                while True:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2 = brightness(img, param_index)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        img2 = Image.fromarray(img2.astype(np.uint8))
                        img2.save(store_path+'brightness_' + str(j) + "_" + image_name + '.JPEG')
                        set_transformed.append(store_path + 'brightness_' + str(j) + "_" + image_name + '.JPEG')
                        #cv2.imwrite(store_path+'_motion_blur_v2_' + str(i) + "_" + image_name + '.png', img2)
                        break

            if transformation == 'contrast':
                img = Image.open(f)
                img_g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                while True:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2 = contrast(img, param_index)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        img2 = Image.fromarray(img2.astype(np.uint8))
                        img2.save(store_path+'contrast_' + str(j) + "_" + image_name + '.JPEG')
                        set_transformed.append(store_path + 'contrast_' + str(j) + "_" + image_name + '.JPEG')
                        #cv2.imwrite(store_path+'_motion_blur_v2_' + str(i) + "_" + image_name + '.png', img2)
                        break

            if transformation == 'jpeg_compression':
                img = Image.open(f)
                img_g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                while True:
                    param_index = random.choice(range(TRANSFORMATION_LEVEL))
                    img2 = jpeg_compression(img, param_index)
                    img2_g = cv2.cvtColor(np.float32(img2), cv2.COLOR_BGR2GRAY)
                    try:
                        if IMAGES == 'cifar-10':
                            IQA_score = eng.vifvec2_layers(
                                matlab.double(np.asarray(img_g).tolist()),
                                matlab.double(np.asarray(img2_g).tolist()))
                        else:
                            IQA_score = eng.vifvec(matlab.double(np.asarray(img_g).tolist()),
                                                   matlab.double(np.asarray(img2_g).tolist()))
                    except:
                        f = random.choice(JPEG_files)
                        continue
                    if 1-IQA_score < t:
                        img2 = Image.fromarray(img2.astype(np.uint8))
                        img2.save(store_path+'jpeg_compression_' + str(j) + "_" + image_name + '.JPEG')
                        set_transformed.append(store_path + 'jpeg_compression_' + str(j) + "_" + image_name + '.JPEG')
                        #cv2.imwrite(store_path+'_motion_blur_v2_' + str(i) + "_" + image_name + '.png', img2)
                        break

    if not os.path.exists('file_lists'):
        os.makedirs('file_lists')

    # write two txt files
    txt = open("file_lists/bootstrap_orig_list_" + save_name + ".txt", "w")
    # print(all_transformed_files)
    # exit()
    for f in set_orig:
        txt.write(f + '\n')
    txt.close()

    txt = open("file_lists/bootstrap_transformed_list_" + save_name + ".txt", "w")
    # print(all_transformed_files)
    # exit()
    for f in set_transformed:
        pwd = os.popen("pwd").read()
        txt.write(pwd.strip() + '/' + f + '\n')
    txt.close()
    eng.quit()
    return set_orig, set_transformed

# obtain detection of the original images


def obtain_orig_detection(YOLO_path, list_orig_path, model, save_filename, run=False):
    if run:
        os.popen("touch " + YOLO_path + "/boot_orig_"+model+'_' + save_filename+".txt")
        print(YOLO_path + "/boot_orig_"+model+'_' + save_filename + ".txt")
        pwd = os.popen("pwd").read()
        print(pwd)
        command = 'cd ' + YOLO_path + '; ./darknet classifier predict cfg/imagenet1k.data cfg/' + model + '.cfg ' + \
            model + '.weights < ' + pwd.strip() + '/' + list_orig_path + ' > boot_orig_'+model+'_' + save_filename+'.txt'
        print(command)
        cmd_result = os.popen(command).read()

    #results_f = open(YOLO_path + '/boot_orig_'+model+'.txt', "r")
    # if run = False, read existing
    if not run:
        results_f = open('bootstrap_results/boot_orig_'+model+'_' + save_filename+'.txt', "r")
    else:
        results_f = open(YOLO_path + 'boot_orig_'+model+'_' + save_filename+'.txt', "r")
    file_name_f = open(pwd.strip() + '/' + list_orig_path, 'r')

    results = {}
    while True:
        line = results_f.readline()
        if not line:
            break
        if line.startswith('Enter Image Path: '):
            filename = re.findall('(ILSVRC2012_val_\d+)', file_name_f.readline())  # [0]
            if filename != []:
                # print(line)
                while re.findall('([^:]+): \d+\.\d+', line) == []:
                    line = results_f.readline()
                matches = re.findall('([^:]+): \d+\.\d+', line)
                detection = matches[0].strip()
                if detection != '':
                    results[filename[0]] = detection
    pickle.dump(results, open('boot_orig_'+model+'_' + save_filename+'.pkl', 'wb'))
    return results

# obtain detection of transformed images


def obtain_transformed_detection(YOLO_path, list_transformed_path, model, save_filename, run=False):
    if run:
        os.popen("touch " + YOLO_path + "/boot_transformed_"+model + '_'+save_filename+".txt")
        print(YOLO_path + "/boot_transformed_"+model + '_'+save_filename+".txt")
        pwd = os.popen("pwd").read()
        print(pwd)
        command = 'cd ' + YOLO_path + '; ./darknet classifier predict cfg/imagenet1k.data cfg/' + model + '.cfg ' + model + \
            '.weights < ' + pwd.strip() + '/' + list_transformed_path + ' > boot_transformed_'+model + '_'+save_filename+'.txt'
        cmd_result = os.popen(command).read()

    #results_f = open(YOLO_path + '/boot_transformed_'+model+'.txt', "r")
    if not run:
        results_f = open('bootstrap_results/boot_transformed_'+model + '_'+save_filename+'.txt', "r")
    else:
        results_f = open(YOLO_path + 'boot_transformed_'+model + '_'+save_filename+'.txt', "r")
    file_name_f = open(pwd.strip() + '/' + list_transformed_path, 'r')
    print(pwd.strip() + '/file_lists/' + list_transformed_path)
    print(YOLO_path + 'boot_transformed_'+model + '_'+save_filename+'.txt')
    results = {}
    while True:
        line = results_f.readline()
        if not line:
            break
        if line.startswith('Enter Image Path: '):
            filename = file_name_f.readline()  # [0]
            if filename != []:
                while re.findall('([^:]+): \d+\.\d+', line) == []:
                    line = results_f.readline()
                    if not line:
                        break
                if not line:
                    break
                matches = re.findall('([^:]+): \d+\.\d+', line)
                detection = matches[0].strip()
                if detection != '':
                    results[filename.strip()] = detection
    pickle.dump(results, open('boot_transformed_'+model+'_' + save_filename+'.pkl', 'wb'))
    return results

# read result files and estimate confidence interval and print metrics


def read_result(result_txt, image_list):
    # find list to label
    list_to_label = "synset_words.txt"
    dict_list_to_label = {}
    f = open(list_to_label, "r")
    for line in f:
        match = re.findall('(n\d+) (.*)', line)
        if match[0][0] not in dict_list_to_label.keys():
            dict_list_to_label[match[0][0]] = []
        dict_list_to_label[match[0][0]] += match[0][1].split(', ')
    f.close()

    ground_truth = {}  # need to obtain from synset_words.txt
    image_list_f = open(image_list, "r")
    image_names_set = []
    while True:
        # Get next line from file
        line = image_list_f.readline()
        if not line:
            break
        # this may change based on the file we are running
        matches = re.findall('(ILSVRC2012\_val\_\d+)\.(n\d+)', line)  # validation
        # others check later, include batch dir for batch results
        filename = matches[0][0]
        ground_truth_list = matches[0][1]
        image_names_set.append(filename)
        ground_truth[filename] = dict_list_to_label[ground_truth_list]
        #print("ground truth: " + filename + " -> " + str(dict_list_to_label[ground_truth_list]))

    image_list_f.close()
    results = {}
    results_f = open(result_txt, "r")
    cur_id = 0
    while True:
        line = results_f.readline()
        if not line:
            break
        if line.startswith('Enter Image Path:'):
            # print(line)
            matches = re.findall('([a-zA-Z\s]+$)', line)
            detection = matches[0].strip()
            if detection != '':
                results[image_names_set[cur_id]] = detection
                #print("detection " + str(cur_id) +" : " + image_names_set[cur_id] + " -> " + detection)
                cur_id += 1
    # for image in ground_truth.keys():
    #	print(image + " ground truth: " + str(ground_truth[image]) + "; detection: " + results[image])
    return ground_truth, results


def calculate_confidence(acc_list, base_acc, req_acc):
    # fitting a normal distribution
    _, bins, _ = plt.hist(acc_list, 20, alpha=0.5, density=True)
    mu, sigma = scipy.stats.norm.fit(acc_list)
    best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
    print("Estimated mean from bootstrapping: " + str(mu))
    print("Estimated sigma from bootstrapping: " + str(sigma))
    exit()
    # calculate the requirement
    #requirement_acc = base_acc - x_squared_inverse(IQA_value)
    #print("requirement is " + str(requirement_acc))
    # print(bins)
    #plt.plot(bins, best_fit_line, label="Normal Distribution")
    #plt.axvline(x=req_acc, label="Accuracy from the Requirement")
    # plt.legend()
    # plt.show()

    distribution = scipy.stats.norm(mu, sigma)
    # if base_acc < mu:
    #	result = distribution.cdf(base_acc)
    # else:
    result = 1 - distribution.cdf(req_acc)
    # print(distribution.cdf(requirement_acc))
    print('confidence of satisfication:' + str(result))
    if result >= 0.5:
        print("requirement SATISFIED")
    else:
        print("requirement NOT SATISFIED")
    return result


def estimate_conf_int(ground_truth, orig_results, transformed_result, a, rq_type):  # rel or abs
    car_labels = ['beach wagon', 'station wagon', 'wagon', 'estate car', 'beach waggon',
                  'station waggon', 'waggon', 'convertible', 'sports car', 'sport car']

    if rq_type == 'abs':
        batch_results_abs = {}
        for trans_img in transformed_result.keys():  # trans_img is transformed image path
            matches = re.findall('(batch\_\d+).*(ILSVRC2012_val_\d+)', trans_img)
            orig_img_name = matches[0][1]       # get original_image
            batch = matches[0][0]               # get batch number
            if batch not in batch_results_abs.keys():
                batch_results_abs[batch] = []           # fill batch_results_abs with a empty dict
            # add original name and transformed prediction pair
            batch_results_abs[batch].append((orig_img_name, transformed_result[trans_img]))
        batch_accuracies = []
        for batch in batch_results_abs.keys():
            batch_accuracies.append(
                sum([1 for x in batch_results_abs[batch] if(x[1] in car_labels) == (ground_truth[x[0]] in car_labels)]) /
                len(batch_results_abs[batch]))
        base_acc = sum([1 for x in orig_results.keys() if (orig_results[x] in car_labels)
                       == (ground_truth[x] in car_labels)])/len(orig_results.keys())
        print("--------------------------------------------")
        print("Verifying Absolute Requirement: ")
        conf_abs = calculate_confidence(batch_accuracies, base_acc, base_acc)  # abs
        print("--------------------------------------------")

        return conf_abs

    if rq_type == 'rel':
        batch_results_rel = {}
        # sort batch results
        for trans_img in transformed_result.keys():
            # print(trans_img)
            matches = re.findall('(batch\_\d+).*(ILSVRC2012_val_\d+)', trans_img)
            orig_img_name = matches[0][1]
            batch = matches[0][0]
            if batch not in batch_results_rel.keys():
                batch_results_rel[batch] = []
            batch_results_rel[batch].append(orig_img_name, transformed_result[trans_img])
        batch_preserved = []
        for batch in batch_results_rel.keys():
            batch_preserved.append(sum([1 for x in batch_results_rel[batch].keys() if (
                batch_results_rel[batch][x] in car_labels) == (orig_results[x] in car_labels)])/len(batch_results_rel[batch].keys()))

        print("--------------------------------------------")
        print(" Verifying Relative Requirement:")
        conf_rel = calculate_confidence(batch_preserved, base_acc, a)  # rel
        print("--------------------------------------------")

        return conf_rel


def print_metrics(ground_truth, detections):
    car_labels = ['beach wagon', 'station wagon', 'wagon', 'estate car', 'beach waggon',
                  'station waggon', 'waggon', 'convertible', 'sports car', 'sport car']
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for image in detections.keys():

        if ground_truth[image][0] in car_labels:
            if detections[image] in car_labels:  # true positive
                true_pos += 1
            else: 	# false negative
                false_neg += 1
        else:  # negatives
            if detections[image] in car_labels:  # false positive
                false_pos += 1
            else: 	# true negative
                true_neg += 1
    accuracy = (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
    #print("Accuracy: " + str(accuracy))
    precision = true_pos/(true_pos+false_pos)
    print("Precision: " + str(precision))
    recall = true_pos/(true_pos+false_neg)
    #print("Recall: " + str(recall))
    print("f1: " + str(2*(recall * precision) / (recall + precision)))
    return accuracy, 2*(recall * precision) / (recall + precision)


def print_metrics_batch_result(ground_truth, batch_result):
    car_labels = ['beach wagon', 'station wagon', 'wagon', 'estate car', 'beach waggon',
                  'station waggon', 'waggon', 'convertible', 'sports car', 'sport car']
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for image in batch_result.keys():
        image_name = re.findall('(ILSVRC2012_val_\d+)', image)[0]
        gt = ground_truth[image_name][0]
        if gt in car_labels:
            if batch_result[image] in car_labels:  # true positive
                true_pos += 1
            else: 	# false negative
                false_neg += 1
        else:  # negatives
            if batch_result[image] in car_labels:  # false positive
                false_pos += 1
            else: 	# true negative
                true_neg += 1
    accuracy = (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
    #print("Accuracy: " + str(accuracy))
    precision = true_pos/(true_pos+false_pos)
    print("Precision: " + str(precision))
    recall = true_pos/(true_pos+false_neg)
    #print("Recall: " + str(recall))
    print("f1: " + str(2*(recall * precision) / (recall + precision)))
    return accuracy, 2*(recall * precision)/(recall + precision)


def read_ground_truth(image_list):
    # find list to label
    list_to_label = "synset_words.txt"
    dict_list_to_label = {}
    f = open(list_to_label, "r")
    for line in f:
        match = re.findall('(n\d+) (.*)', line)
        if match[0][0] not in dict_list_to_label.keys():
            dict_list_to_label[match[0][0]] = []
        dict_list_to_label[match[0][0]] += match[0][1].split(', ')
    f.close()

    ground_truth = {}  # need to obtain from synset_words.txt
    image_list_f = open(image_list, "r")
    image_names_set = []
    while True:
        # Get next line from file
        line = image_list_f.readline()
        if not line:
            break
        matches = re.findall('(ILSVRC2012\_val\_\d+)\.(n\d+)', line)  # validation
        filename = matches[0][0]
        ground_truth_list = matches[0][1]
        image_names_set.append(filename)
        ground_truth[filename] = dict_list_to_label[ground_truth_list]
    '''
	image_list_f.close()
	results = {}
	results_f = open(result_txt, "r")
	cur_id = 0
	while True:
		line = results_f.readline()
		if not line:
			break
		if line.startswith('Enter Image Path:'):
			matches = re.findall('([a-zA-Z\s]+$)', line)
			detection = matches[0].strip()
			if detection != '':
				results[image_names_set[cur_id]] = detection
				cur_id += 1
	'''
    return ground_truth  # , results


def usage():
    #usage = ""
    parser = argparse.ArgumentParser(
        description='''My Description. And what a lovely description it is. ''',
        epilog="""All is well that ends well.""")
    # parser.add_argument('--foo', type=int，help='FOO!')
    parser.add_argument('-d', '--Darknet_path', type=str, help='path to Darknet directory')
    parser.add_argument('-r', '--read_existing', type=str, help='read from exisitng run')
    parser.add_argument('-i', '--image_path', type=str, help='path to ILSVRC2012 validation images')
    parser.add_argument('-n', '--number_of_batches', type=int,
                        help='the number of batches for bootstrapping, must be integer')
    parser.add_argument('-s', '--batch_size', type=int,
                        help='the number of images per batch for bootstrapping, must be integer')
    #parser.add_argument('bar', nargs='*', default=[1, 2, 3], help='BAR!')
    args = parser.parse_args()


    # return usage
if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hd:ri:n:s:", ["help", "Darknet_path=",
                                   "read_existing", "image_path", "number_of_batches", "batch_size"])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        # usage()
        sys.exit(2)
    YOLO_path = None
    read_exist = False
    image_path = None
    number_of_batches = 200
    batch_size = 500
    for o, a in opts:
        if o in ("-r", "--read_existing"):  # o == "-v":
            read_exist = True
            print("reading exising")
        if o in ("-h", "--help"):
            usage()
            print("help")
            sys.exit()
        if o in ("-d", "--Darknet_path"):
            YOLO_path = a
            if not YOLO_path.endswith('/'):
                YOLO_path += '/'
            print("YOLO_path" + YOLO_path)
        if o in ("-i", "--image_path"):
            image_path = a
            if not image_path.endswith('/'):
                image_path += '/'
            print("image_path" + image_path)
        if o in ("-n", "--number_of_batches"):
            try:
                number_of_batches = int(a)
            except:
                print("invalid number of batches, must be integer")
        if o in ("-s", "--batch_size"):
            try:
                batch_size = int(a)
            except:
                print("invalid batch size, must be integer")

    ground_truth = read_ground_truth('inet.val.list')
    satisfaction_scores = {}
    for trans in list_transformations_G:
        satisfaction_scores[trans] = {}

        for rq in ['abs', 'rel']:
            save_name = trans + '_' + rq
            if not read_exist and not image_path:
                print("missing path to validation images")
                sys.exit()
            if not read_exist:
                # if trans != 'Blur':
                #	t = t_params['generalized'][rq]
                # else:
                t = t_params[trans][rq]
                #gen_bootstrapping(number_of_batches, image_path,  'bootstrap_samples/', t, save_name, batch_size=batch_size, transformation=trans)
                # continue

            models = ['darknet19', 'darknet53_448', 'alexnet', 'resnext50', 'vgg-16', 'resnet50']

            print("###################" + rq + "###################")
            for model in models:
                if read_exist:
                    orig_results = obtain_orig_detection(
                        YOLO_path, "file_lists/bootstrap_orig_list_" + save_name + ".txt", model, rq+'_'+model, run=False)
                    transformed_results = obtain_transformed_detection(
                        YOLO_path, "bootstrap_results/bootstrap_transformed_list_" + save_name + ".txt", model, rq+'_'+model, run=False)
                else:
                    orig_results = obtain_orig_detection(
                        YOLO_path, "file_lists/bootstrap_orig_list_" + save_name + ".txt", model, rq+'_'+model, run=True)
                    print(orig_results)

                    transformed_results = obtain_transformed_detection(
                        YOLO_path, "file_lists/bootstrap_transformed_list_" + save_name + ".txt", model, rq+'_'+model, run=True)
                    print(transformed_results)
                    # exit()
                '''
				print("ii. Precision and f1 on sampled transformed images from bootstrap:")
				accuracy, f1 = print_metrics_batch_result(ground_truth, transformed_results)
				trans_acc.append(accuracy)
				trans_f1.append(f1)
				'''
                conf = estimate_conf_int(ground_truth, orig_results, transformed_results, 0.95, rq)
                satisfaction_scores[trans][rq] = conf
                print(satisfaction_scores)
