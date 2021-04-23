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
from scipy.ndimage.filters import gaussian_filter
from scipy import fftpack as fp
from skimage.color import rgb2gray, rgb2grey

# sample transformation parameter values with in the specified range, here chosen to be evenly distributed
# obtained through sampling images and checking the vd value in the requirements
intensity_shift_params = list(range(-120, 121))
gaussian_noise_params = list(range(4, 49))
gamma_params = [x/100 for x in list(range(90, 109))]
contrast_params = [x/10 for x in list(range(1, 10))]
uniform_noise_params = [x/10 for x in list(range(0, 7))]
lowpass_params = [x/10 for x in list(range(0, 30))]
highpass_params = [x/10 for x in list(range(5, 150))]
phase_noise_params = list(range(0, 90))
# transformations
def intensity_shift(img, degree, precision=0):  
	return (img + np.around(degree, precision)).clip(min=0, max=255) 

def gaussian_noise(img, state, img_h, img_w, ch, precision=0): 
	# currently gaussian with mean 0
	if ch > 1:
		gauss = np.random.normal(0,state,(img_h, img_w, ch))
	else:
		gauss = np.random.normal(0,state,(img_h, img_w))
	noisy = img + np.around(gauss, precision)
	return noisy.clip(min=0, max=255) 

def adjust_gamma(img, gamma, precision=0):
	return np.around(exposure.adjust_gamma(img, gamma), precision)

def adjust_contrast(image, contrast_level):
	"""
	Taken from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/code/image_manipulation.py
	"""

	assert(contrast_level >= 0.0), "contrast_level too low."
	assert(contrast_level <= 1.0), "contrast_level too high."

	return (1-contrast_level)/2.0 + image.dot(contrast_level)

def apply_uniform_noise(image, low, high, rng=None):
	"""
	Taken from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/code/image_manipulation.py
	"""    
	nrow = image.shape[0]
	ncol = image.shape[1]
	nch = image.shape[2]   
	
	image = image / 255 
	
	image = image + get_uniform_noise(low, high, nrow, ncol, nch, rng)    #clip values
	
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
    #image = rgb2grey(image)
    new_image = np.zeros(image.shape, image.dtype)

    # aplly Gaussian low-pass filter
    new_image = gaussian_filter(image, std, mode ='constant', cval=bg_grey)

    # crop too small and too large values
    #new_image[new_image < 0] = 0
    #new_image[new_image > 1] = 1

    # return stacked (RGB) grey image
    #return np.dstack((new_image,new_image,new_image))
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
    #image = rgb2grey(image)
    new_image = np.zeros(image.shape, image.dtype)

    # aplly the gaussian filter and subtract from the original image
    gauss_filter = gaussian_filter(image, std, mode ='constant', cval=bg_grey)
    new_image = image - gauss_filter

    # add mean of old image to retain image statistics
    mean_diff = bg_grey - np.mean(new_image, axis=(0,1))
    new_image = new_image + mean_diff

    # crop too small and too large values
    #new_image[new_image < 0] = 0
    #new_image[new_image > 1] = 1

    # return stacked (RGB) grey image
    #return np.dstack((new_image,new_image,new_image))
    new_image = new_image * 255
    return new_image.clip(min=0, max=255) 

def scramble_phases(image, width):
    """
    Taken from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/code/image_manipulation.py
    """
    image = image / 255 
    # create array with random phase shifts from the interval [-width,width]
    length = (image.shape[0]-1)*(image.shape[1]-1)
    phase_shifts = np.random.random(length//2) - 0.5
    phase_shifts = phase_shifts * 2 * width/180 * np.pi
    
    # convert to graysclae
    channel = rgb2grey(image)
    #channel = image
    #print(channel.shape)

    # Fourier Forward Tranform and shift to centre
    f = fp.fft2(channel) #rfft for real values
    f = fp.fftshift(f)


    # get amplitudes and phases
    f_amp = np.abs(f)
    f_phase = np.angle(f)

    # transformations of phases
    # just change the symmetric parts of FFT outcome, which is
    # [1:,1:] for the case of even image sizes
    fnew_phase = f_phase
    #print(f_phase.shape)
    #print(phase_shifts.shape)
    fnew_phase[1:,1:] = shift_phases(f_phase[1:,1:], phase_shifts)

    # recalculating FFT complex representation from new phases and amplitudes
    fnew = f_amp*np.exp(1j*fnew_phase)

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
    print(phase_shifts.shape)
    print(flat_phase[:length//2].shape)
    if length%2 == 1:
        flat_phase[:length//2] += phase_shifts
        flat_phase[length//2+1:] -= phase_shifts
    else:
        flat_phase[:length//2] += phase_shifts
        flat_phase[length//2:] -= phase_shifts


    # reshape into output format
    f_phase = flat_phase.reshape(f_shape)

    return f_phase



def filter_image_classes(JPEG_files, object_class):
	class_to_list = 'MSCOCO_to_ImageNet_category_mapping.txt'
	dict_class_to_list = {}
	f = open(class_to_list, "r")
	for line in f:
		match = re.findall('\s+(.*)\s+=\s+\[(.*)\]', line)
		if len(match) > 0:
			if match[0][0] not in dict_class_to_list.keys():
				dict_class_to_list[match[0][0]] = []
			dict_class_to_list[match[0][0]] += match[0][1].split(', ')
	f.close()
	all_lists = sum(dict_class_to_list.values(),[])# ' '.join(dict_class_to_list.values())
	#mapping_file = open(class_to_list, "r").readlines()
	#labels = re.findall('(n\d+)', ' '.join(mapping_file))
	#print(len(JPEG_files))
	def find_label(filename, object_class):
		label = re.findall('\.(n\d+)\.', filename)[0]
		if label in dict_class_to_list[object_class]:
			return True
		else:
			x = random.uniform(0, 1)#x = random()500/(10350 - 150)
			if x < 100/(10350 - 150):
				return True
			else:
				return False
	JPEG_files_car = [f.strip() for f in JPEG_files if re.findall('\.(n\d+)\.', f)[0] in dict_class_to_list['car']]
	JPEG_files_not_car = random.sample([f.strip() for f in JPEG_files if re.findall('\.(n\d+)\.', f)[0] not in dict_class_to_list['car']], 500)
	print(len(JPEG_files_car))
	print(len(JPEG_files_not_car))
	return JPEG_files_car + JPEG_files_not_car


def gen_bootstrapping(num_batch, orig_path, gen_path, object_class, batch_size=50, transformation='intensity_shift'):
	if not os.path.exists(gen_path):
		os.makedirs(gen_path)
	JPEG_files = open(orig_path, "r").readlines()#[y for x in os.walk(orig_path) for y in glob(os.path.join(x[0], '*.JPEG'))]
	JPEG_files = filter_image_classes(JPEG_files, object_class)
	#print(JPEG_files)
	#exit()
	set_orig = []
	set_transformed = []
	pwd = os.popen("pwd").read().strip()
	for i in range(num_batch):
		print("batch " + str(i))
		batch = random.sample(JPEG_files, batch_size)
		set_orig += batch
		if not os.path.exists(gen_path + "batch_" +str(i) + '/'):
			os.makedirs(gen_path + "batch_" +str(i) + '/')
		store_path = gen_path + "batch_" +str(i) + '/'
		for f in batch:
			image_name = re.findall('(ILSVRC2012_val_\d+)', f)[0]
			img = np.asarray(cv2.imread(f), dtype=np.float32)
			if transformation == 'gaussian_noise':
				img_h, img_w, ch = img.shape
				param = random.choice(gaussian_noise_params)
				img2 = gaussian_noise(img, param, img_h, img_w, ch)
				cv2.imwrite(pwd + '/' + store_path+'gaussian_' + str(param) + "_" + image_name + '.JPEG', img2)
				set_transformed.append(pwd + '/' + store_path+ 'gaussian_' + str(param) + "_" + image_name + '.JPEG')
			if transformation == 'intensity_shift':
				param = random.choice(intensity_shift_params)
				img2 = intensity_shift(img, param)
				cv2.imwrite(pwd  + '/'+ store_path+'intensity_' + str(param) + "_" + image_name + '.JPEG', img2)
				set_transformed.append(pwd + '/'+ store_path+ 'intensity_' + str(param) + "_" + image_name + '.JPEG')
				#print(pwd  + '/'+ store_path+'intensity_' + str(param) + "_" + image_name + '.JPEG')
				#exit()
			if transformation == 'gamma':
				param = random.choice(gamma_params)
				img2 = adjust_gamma(img, param)
				cv2.imwrite(pwd  + '/'+ store_path+'gamma_' + str(param) + "_" + image_name + '.JPEG', img2)
				set_transformed.append(pwd  + '/'+ store_path+ 'gamma_' + str(param) + "_" + image_name + '.JPEG')

			if transformation == 'contrast':
				param = random.choice(contrast_params)
				img2 = adjust_contrast(img, param)
				cv2.imwrite(pwd  + '/'+ store_path+'contrast_' + str(param) + "_" + image_name + '.JPEG', img2)
				set_transformed.append(pwd  + '/'+ store_path+ 'contrast_' + str(param) + "_" + image_name + '.JPEG')
			
			if transformation == 'uniform_noise':
				param = random.choice(uniform_noise_params)
				img2 = apply_uniform_noise(img, 0, param)
				cv2.imwrite(pwd  + '/'+ store_path+'uniform_noise_' + str(param) + "_" + image_name + '.JPEG', img2)
				set_transformed.append(pwd  + '/'+ store_path+ 'uniform_noise_' + str(param) + "_" + image_name + '.JPEG')

			if transformation == 'lowpass':
				param = random.choice(lowpass_params)
				img2 = low_pass_filter(img, param)
				cv2.imwrite(pwd  + '/'+ store_path+'lowpass_' + str(param) + "_" + image_name + '.JPEG', img2)
				set_transformed.append(pwd  + '/'+ store_path+ 'lowpass_' + str(param) + "_" + image_name + '.JPEG')

			if transformation == 'highpass':
				param = random.choice(highpass_params)
				img2 = high_pass_filter(img, param)
				cv2.imwrite(pwd  + '/'+ store_path+'highpass_' + str(param) + "_" + image_name + '.JPEG', img2)
				set_transformed.append(pwd  + '/'+ store_path+ 'highpass_' + str(param) + "_" + image_name + '.JPEG')

			if transformation == 'phase_noise':
				param = random.choice(phase_noise_params)
				img2 = scramble_phases(img, param)
				cv2.imwrite(pwd  + '/'+ store_path+'phase_noise_' + str(param) + "_" + image_name + '.JPEG', img2)
				set_transformed.append(pwd  + '/'+ store_path+ 'phase_noise_' + str(param) + "_" + image_name + '.JPEG')

	# write two txt files
	txt = open("bootstrap_orig_list.txt", "w")
	for f in set_orig:
		txt.write(f + '\n')
	txt.close()

	txt = open("bootstrap_transformed_list.txt", "w")
	for f in set_transformed:
		txt.write(f + '\n')
	txt.close()
	return set_orig, set_transformed

# obtain detection of the original images
def obtain_orig_detection(YOLO_path, list_orig_path, model_detection, run=False):
	if run:
		os.popen("touch " + YOLO_path +"/boot_orig_"+model+ ".txt")
		pwd = os.popen("pwd").read()
		print(pwd)
		command = 'cd ' + YOLO_path + '; ./darknet classifier predict cfg/imagenet1k.data cfg/' + model +'.cfg '+ model + '.weights < ' +pwd.strip() + '/' + list_orig_path + ' > boot_orig_'+model+'.txt'
		print(command)
		cmd_result = os.popen(command).read()

	#if not run:
	#	results_f = open('bootstrap_results/boot_orig_'+model+'.txt', "r")
	#else:
	#	results_f = open(YOLO_path +'boot_orig_'+model+'.txt', "r")
	results_f = open(model_detection, "r")
	file_name_f = open(list_orig_path, 'r')
	results = {}
	while True:
		line = results_f.readline()
		if not line:
			break
		if line.startswith('Enter Image Path:'):
			filename = re.findall('(ILSVRC2012_val_\d+)', file_name_f.readline())#[0]
			if filename != []:
				matches = re.findall('([a-zA-Z\s]+$)', line)
				detection = matches[0].strip()
				if detection != '':
					results[filename[0]] = detection
	return results

# obtain detection of transformed images
def obtain_transformed_detection(YOLO_path, list_transformed_path, model_detection, run=False):
	if run:
		os.popen("touch " + YOLO_path +"/boot_transformed_"+model+ ".txt")
		pwd = os.popen("pwd").read()
		command = 'cd ' + YOLO_path + '; ./darknet classifier predict cfg/imagenet1k.data cfg/' + model +'.cfg '+ model + '.weights < ' +pwd.strip() + '/' + list_transformed_path + ' > boot_transformed_'+model+'.txt'
		cmd_result = os.popen(command).read()

	#if not run:
	#	results_f = open('bootstrap_results/boot_transformed_'+model+'.txt', "r")
	#else:
	#	results_f = open(YOLO_path +'boot_transformed_'+model+'.txt', "r")
	results_f = open(model_detection, "r")
	file_name_f = open(list_transformed_path, 'r')
	results = {}
	while True:
		line = results_f.readline()
		if not line:
			break
		if line.startswith('Enter Image Path:'):
			filename = file_name_f.readline()
			matches = re.findall('([a-zA-Z\s]+$)', line)
			detection = matches[0].strip()
			if detection != '':
				results[filename.strip()] = detection
	return results

# read result files and estimate confidence interval and print metrics
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

	ground_truth = {} # need to obtain from synset_words.txt
	image_list_f = open(image_list, "r")
	image_names_set = []
	while True:
	# Get next line from file
		line = image_list_f.readline()
		if not line:
			break
		matches = re.findall('(ILSVRC2012\_val\_\d+)\.(n\d+)', line) #validation
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
	return ground_truth#, results

def calculate_confidence(acc_list, base_acc, req_acc):
	# fitting a normal distribution
	#print(acc_list)
	_, bins, _ = plt.hist(acc_list, 30, alpha=0.5, density=True)
	mu, sigma = scipy.stats.norm.fit(data=acc_list)
	best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
	#print("Estimated mean from bootstrapping: " + str(round(mu,3)))
	#print("Estimated sigma from bootstrapping: " + str(round(sigma,3)))
	# calculate the requirement
	#requirement_acc = base_acc - x_squared_inverse(IQA_value) 
	#print("requirement is " + str(requirement_acc))
	#print(bins)
	#plt.plot(bins, best_fit_line, label="Normal Distribution")
	#plt.axvline(x=req_acc, label="Accuracy from the Requirement")
	#plt.legend()
	#plt.show()
	
	distribution = scipy.stats.norm(mu, sigma)
	result = 1- distribution.cdf(req_acc)
	#print('confidence of satisfication:' + str(round(result,3)))
	#if result >= 0.5:
	#	print("requirement SATISFIED")
	#else:
	#	print("requirement NOT SATISFIED")
	#return result
	#print(round(mu - req_acc, 3))
	return round(result,3), round(mu,3), round(sigma,3)

def estimate_conf_int(ground_truth, orig_results, transformed_result, a, labels): #rel or abs
	batch_results_abs = {}
	batch_results_rel = {}
	#sort batch results
	for trans_img in transformed_result.keys():
		matches = re.findall('(batch\_\d+).*(ILSVRC2012_val_\d+)', trans_img)
		orig_img_name = matches[0][1]
		batch = matches[0][0]
		if batch not in batch_results_abs.keys():
			batch_results_abs[batch] = {}
		if batch not in batch_results_rel.keys():
			batch_results_rel[batch] = {}

		#if ground_truth[orig_img_name][0] in labels:
		batch_results_abs[batch][orig_img_name] = transformed_result[trans_img]
		#if transformed_result[trans_img] in labels:
		batch_results_rel[batch][orig_img_name] = transformed_result[trans_img]

	# obtain abs
	batch_accuracies = []
	for batch in batch_results_abs.keys():
		if len(batch_results_abs[batch].keys()) > 0:
			batch_accuracies.append(sum([1 for x in batch_results_abs[batch].keys() if (batch_results_abs[batch][x] in labels) == (ground_truth[x][0] in labels)])/len(batch_results_abs[batch].keys()))

	batch_preserved = []
	for batch in batch_results_rel.keys():
		if len(batch_results_rel[batch].keys()) > 0:
			batch_preserved.append(sum([1 for x in batch_results_rel[batch].keys() if (batch_results_rel[batch][x] in labels) == (orig_results[x] in labels)] )/len(batch_results_rel[batch].keys()))

	#print("ABS batches")
	#print(batch_accuracies)
	#print("REL batches")
	#print(batch_preserved)

	base_acc = sum([1 for x in orig_results.keys() if (orig_results[x] in labels) == (ground_truth[x][0] in labels) ])/len(orig_results.keys())
	#print("--------------------------------------------")
	#print("1. Verifying Absolute Requirement: ")
	conf_abs, mu_abs, sigma_abs = calculate_confidence(batch_accuracies, base_acc, base_acc) #abs
	#print("--------------------------------------------")
	#print("2. Verifying Relative Requirement:")
	conf_rel, mu_rel, sigma_rel = calculate_confidence(batch_preserved, base_acc, a) #rel
	#print("--------------------------------------------")

	return conf_abs, mu_abs, sigma_abs, conf_rel, mu_rel, sigma_rel

def print_metrics(ground_truth, detections, labels):
	true_pos = 0
	true_neg = 0
	false_pos = 0
	false_neg = 0
	for image in detections.keys():
		if ground_truth[image][0] in labels: 
			if detections[image] in labels: # true positive
				true_pos += 1
			else: 	# false negative
				false_neg += 1
		else: #negatives
			if detections[image] in labels: # false positive
				false_pos += 1
			else: 	# true negative
				true_neg += 1
	accuracy = (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
	accuracy = round(accuracy, 3)
	print("Accuracy: " + str(accuracy))
	precision = true_pos/(true_pos+false_pos)
	#print("Precision: " + str(precision))
	recall = true_pos/(true_pos+false_neg)
	#print("Recall: " + str(recall))
	f1  = 2*(recall * precision) / (recall + precision)
	f1 = round(f1, 3)
	print("f1: " + str(f1))
	return accuracy, f1

def print_metrics_batch_result(ground_truth, batch_result, labels):
	true_pos = 0
	true_neg = 0
	false_pos = 0
	false_neg = 0
	for image in batch_result.keys():
		image_name = re.findall('(ILSVRC2012_val_\d+)', image)[0]
		gt = ground_truth[image_name][0]
		if gt in labels: 
			if batch_result[image] in labels: # true positive
				true_pos += 1
			else: 	# false negative
				false_neg += 1
		else: #negatives
			if batch_result[image] in labels: # false positive
				false_pos += 1
			else: 	# true negative
				true_neg += 1
	accuracy = (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
	accuracy = round(accuracy, 3)
	print("Accuracy: " + str(accuracy))
	precision = true_pos/(true_pos+false_pos)
	#print("Precision: " + str(precision))
	recall = true_pos/(true_pos+false_neg)
	#print("Recall: " + str(recall))
	f1 = 2*(recall * precision) / (recall + precision)
	f1 = round(f1, 3)
	print("f1: " + str(f1))
	return accuracy, f1

def print_ML_metrics(ground_truth, orig_detections, batch_result, labels):
	true_pos = 0
	true_neg = 0
	false_pos = 0
	false_neg = 0

	for image in orig_detections.keys():
		if ground_truth[image][0] in labels: 
			if orig_detections[image] in labels: # true positive
				true_pos += 1
			else: 	# false negative
				false_neg += 1
		else: #negatives
			if orig_detections[image] in labels: # false positive
				false_pos += 1
			else: 	# true negative
				true_neg += 1

	for image in batch_result.keys():
		image_name = re.findall('(ILSVRC2012_val_\d+)', image)[0]
		gt = ground_truth[image_name][0]
		if gt in labels: 
			if batch_result[image] in labels: # true positive
				true_pos += 1
			else: 	# false negative
				false_neg += 1
		else: #negatives
			if batch_result[image] in labels: # false positive
				false_pos += 1
			else: 	# true negative
				true_neg += 1

	accuracy = (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
	accuracy = round(accuracy, 3)
	#print("Accuracy: " + str(accuracy))
	precision = true_pos/(true_pos+false_pos)
	#print("Precision: " + str(precision))
	recall = true_pos/(true_pos+false_neg)
	#print("Recall: " + str(recall))
	f1 = 2*(recall * precision) / (recall + precision)
	f1 =  round(f1, 3)
	#print("f1: " + str(f1))
	return accuracy, f1


def usage():
	parser=argparse.ArgumentParser(
	description='''This is the script for our subprocess IV Verifying MLC against requirement ''')
	parser.add_argument('-d', '--Darknet_path', type=str, help='path to Darknet directory')
	parser.add_argument('--load_existing', type=str, help='to load the experiment results from the paper for the transformation specified by -t')
	parser.add_argument('-r', type=str, help='read from previous run result')
	parser.add_argument('-i', '--image_name_file', type=str, help='path to inet.val.list')
	parser.add_argument('-n', '--number_of_batches', type=int, help='the number of batches for bootstrapping, must be integer, default is 200')
	parser.add_argument('-s', '--batch_size', type=int, help='the number of images per batch for bootstrapping, must be integer, default is 500')
	parser.add_argument('-c', '--class', type=str, help='the object class y to test in both requirements, default is car, must be one of airplane, bicycle, boat, car, chair, dog, keyboard, oven, bear, bird, bottle, cat, clock, elephant, knife, truck')
	parser.add_argument('-t', '--transformation', type=str, help='the transformation to test, choose from one of intensity_shift, gaussian_noise, gamme, contrast, uniform_noise, lowpass, highpass and phase_noise, default is intensity_shift')
	args=parser.parse_args()

if __name__ == "__main__":
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hd:i:n:s:c:rt:", ["help", "Darknet_path=", "load_existing", "image_name_file=", "number_of_batches=", "batch_size=", "class=", "transformation="])
	except getopt.GetoptError as err:
		# print help information and exit:
		print(err)  
		usage()
		sys.exit(2)


	list_labels = ['airplane', 'bicycle', 'boat', 'car', 'chair', 'dog', 'keyboard', 'oven', 'bear', 'bird', 'bottle', 'cat', 'clock', 'elephant', 'knife', 'truck']

	class_to_labels = {}
	class_to_labels['airplane'] = ['airliner']
	class_to_labels['bicycle'] = ['bicycle-built-for-two', 'tandem bicycle', 'tandem', 'mountain bike', 'all-terrain bike', 'off-roader']
	class_to_labels['boat'] = ['canoe', 'fireboat', 'lifeboat', 'speedboat', 'yawl']
	class_to_labels['car'] = ['beach wagon', 'station wagon', 'wagon', 'estate car', 'beach waggon', 'station waggon', 'waggon', 'convertible', 'sports car', 'sport car']
	class_to_labels['chair'] = ['barber chair', 'folding chair', 'rocking chair', 'rocker', 'throne']
	class_to_labels['dog'] = ['Japanese spaniel', 'Maltese dog', 'Maltese terrier', 'Maltese', 'Pekinese', 'Pekingese', 'Peke', 'Shih-Tzu', 'Blenheim spaniel', 'papillon', 'toy terrier', 'Rhodesian ridgeback', 'Afghan hound', 'Afghan', 'basset', 'basset hound', 'beagle', 'bloodhound', 'sleuthhound', 'bluetick', 'black-and-tan coonhound', 'Walker hound', 'Walker foxhound', 'English foxhound', 'redbone', 'borzoi', 'Russian wolfhound', 'Irish wolfhound', 'Italian greyhound', 'whippet', 'Ibizan hound', 'Ibizan Podenco', 'Norwegian elkhound', 'elkhound', 'otterhound', 'otter hound', 'Saluki', 'gazelle hound', 'Scottish deerhound', 'deerhound', 'Weimaraner', 'Staffordshire bullterrier', 'Staffordshire bull terrier', 'American Staffordshire terrier', 'Staffordshire terrier', 'American pit bull terrier', 'pit bull terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier', 'Lakeland terrier', 'Sealyham terrier', 'Sealyham', 'Airedale', 'Airedale terrier', 'Australian terrier', 'Dandie Dinmont', 'Dandie Dinmont terrier', 'Boston bull', 'Boston terrier', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer', 'Scotch terrier', 'Scottish terrier', 'Scottie', 'Tibetan terrier', 'chrysanthemum dog', 'silky terrier', 'Sydney silky', 'soft-coated wheaten terrier', 'West Highland white terrier', 'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short-haired pointer', 'vizsla', 'Hungarian pointer', 'English setter', 'Irish setter', 'red setter', 'Gordon setter', 'Brittany spaniel', 'clumber', 'clumber spaniel', 'English springer', 'English springer spaniel', 'Welsh springer spaniel', 'cocker spaniel', 'English cocker spaniel', 'cocker', 'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'komondor', 'Old English sheepdog', 'bobtail', 'Shetland sheepdog', 'Shetland sheep dog', 'Shetland', 'collie', 'Border collie', 'Bouvier des Flandres', 'Bouviers des Flandres', 'Rottweiler', 'German shepherd', 'German shepherd dog', 'German police dog', 'alsatian', 'Doberman', 'Doberman pinscher', 'miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher', 'bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard', 'St Bernard', 'Eskimo dog', 'husky', 'malamute', 'malemute', 'Alaskan malamute', 'Siberian husky', 'affenpinscher', 'monkey pinscher', 'monkey dog', 'basenji', 'pug', 'pug-dog', 'Leonberg', 'Newfoundland', 'Newfoundland dog', 'Great Pyrenees', 'Pomeranian', 'keeshond', 'Brabancon griffon', 'Pembroke', 'Pembroke Welsh corgi', 'toy poodle', 'miniature poodle', 'standard poodle', 'Mexican hairless']
	class_to_labels['keyboard'] = ['computer keyboard', 'keypad', 'typewriter keyboard']
	class_to_labels['oven'] = ['rotisserie']
	class_to_labels['bear'] = ['brown bear', 'bruin', 'Ursus arctos', 'American black bear', 'black bear', 'Ursus americanus', 'Euarctos americanus', 'ice bear', 'polar bear', 'Ursus Maritimus', 'Thalarctos maritimus', 'sloth bear', 'Melursus ursinus', 'Ursus ursinus']
	class_to_labels['bird'] = ['hen', 'brambling', 'Fringilla montifringilla', 'goldfinch', 'Carduelis carduelis', 'house finch', 'linnet', 'Carpodacus mexicanus', 'junco', 'snowbird', 'indigo bunting', 'indigo finch', 'indigo bird', 'Passerina cyanea', 'robin', 'American robin', 'Turdus migratorius', 'bulbul', 'magpie', 'chickadee', 'water ouzel', 'dipper', 'bald eagle', 'American eagle', 'Haliaeetus leucocephalus', 'vulture', 'great grey owl', 'great gray owl', 'Strix nebulosa', 'black grouse', 'ptarmigan', 'ruffed grouse', 'partridge', 'Bonasa umbellus', 'prairie chicken', 'prairie grouse', 'prairie fowl', 'African grey', 'African gray', 'Psittacus erithacus', 'macaw', 'sulphur-crested cockatoo', 'Kakatoe galerita', 'Cacatua galerita', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'red-breasted merganser', 'Mergus serrator', 'goose', 'black swan', 'Cygnus atratus', 'white stork', 'Ciconia ciconia', 'black stork', 'Ciconia nigra', 'spoonbill', 'flamingo', 'little blue heron', 'Egretta caerulea', 'American egret', 'great white heron', 'Egretta albus', 'bittern', 'limpkin', 'Aramus pictus', 'European gallinule', 'Porphyrio porphyrio', 'American coot', 'marsh hen', 'mud hen', 'water hen', 'Fulica americana', 'bustard', 'ruddy turnstone', 'Arenaria interpres', 'red-backed sandpiper', 'dunlin', 'Erolia alpina', 'redshank', 'Tringa totanus', 'dowitcher', 'oystercatcher', 'oyster catcher', 'pelican', 'king penguin', 'Aptenodytes patagonica']
	class_to_labels['bottle'] = ['beer bottle', 'pill bottle', 'pop bottle', 'soda bottle', 'water bottle', 'water jug', 'whiskey jug', 'wine bottle']
	class_to_labels['cat'] = ['tabby', 'tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat', 'Siamese', 'Egyptian cat', 'cougar', 'puma', 'catamount', 'mountain lion', 'painter', 'panther', 'Felis concolor']
	class_to_labels['clock'] = ['analog clock', 'digital clock', 'wall clock']
	class_to_labels['elephant'] = ['Indian elephant', 'Elephas maximus', 'African elephant', 'Loxodonta africana']
	class_to_labels['knife'] = ['cleaver', 'meat cleaver', 'chopper']
	class_to_labels['truck'] = ['fire engine', 'fire truck', 'garbage truck', 'dustcart', 'minivan', 'moving van', 'pickup', 'pickup truck', 'police van', 'police wagon', 'paddy wagon', 'patrol wagon', 'wagon', 'black Maria', 'tow truck', 'tow car', 'wrecker', 'trailer truck', 'tractor trailer', 'trucking rig', 'rig', 'articulated lorry', 'semi']

	list_transf = ['intensity_shift', "gaussian_noise", "gamma", 'contrast', 'uniform_noise', 'lowpass', "highpass", "phase_noise"]

	YOLO_path = None
	read_exist = False
	image_name_file = None
	number_of_batches = 200
	batch_size = 500
	object_class = 'car'
	only_read = False
	load_existing_path = 'bootstrap_results/'
	transf = 'intensity_shift'
	for o,a in opts:
		if o in ("-r"):
			only_read = True
		elif o in ("--load_existing"):
			read_exist = True
		elif o in ("-h", "--help"):
			usage()
			sys.exit()
		elif o in ("-d", "--Darknet_path"):
			YOLO_path = a
			if not YOLO_path.endswith('/'):
				YOLO_path += '/'
		elif o in ("-i", "--image_name_file"):
			image_name_file = a
		elif o in ("-n", "--number_of_batches"):
			try:
				number_of_batches = int(a)
			except:
				print("invalid number of batches, must be integer")
				sys.exit()
		elif o in ("-s", "--batch_size"):
			try:
				batch_size = int(a)
			except:
				print("invalid batch size, must be integer")
				sys.exit()
		elif o in ("-c", "--class"):
			object_class = a
			if object_class not in list_labels:
				print("invalid object class, must be one of airplane, bicycle, boat, car, chair, dog, keyboard, oven, bear, bird, bottle, cat, clock, elephant, knife, truck")
				sys.exit()
		elif o in ('-t', '--transformation'):
			transf = a
			if transf not in list_transf:
				print("transformation must be from one of intensity_shift, gaussian_noise, gamma, contrast, uniform_noise, lowpass, highpass, phase_noise")
				sys.exit()
		else:
			assert False, "unhandled option"
			print("invalid argument\n")
			usage()
			sys.exit()
	if not read_exist and not image_name_file:
		print("missing path to inet.val.list")
		sys.exit()

	
	labels = class_to_labels[object_class]
	
	if not read_exist and not only_read:
		gen_bootstrapping(number_of_batches, image_name_file, 'bootstrap_samples/', object_class, batch_size=batch_size, transformation=transf)

	models = ['alexnet', 'darknet19', 'darknet53_448', 'resnet50', 'resnext50', 'vgg-16']

	l_mu_abs = []
	l_sigma_abs = []
	sat_abs = []
	l_mu_rel = []
	l_sigma_rel = []
	sat_rel = []
	test_acc = []
	test_f1 = []


	print("---------------------------------------" + transf + "----------------------------------------")
	print("|       | verifying accuracy-preserving | verifying prediction-preserving | ML metrics (accuracy, f1)|")
	print('----------------------------------------------------------------------------------------------')
	print('| model |   mu   |   sigma  | conf_acc  |   mu   |   sigma   |  conf_acc  |   accuracy   |     f1    |')
	print('------------------------------------------------------------------------------------------------------')

	for model in models:	
		if not read_exist:
			ground_truth = read_ground_truth(image_name_file)
		else:
			ground_truth = read_ground_truth('bootstrap_results/inet.val.list')

		if read_exist:
			orig_results = obtain_orig_detection(YOLO_path, load_existing_path + transf+"/bootstrap_orig_list.txt", load_existing_path+transf+'/boot_orig_'+model+'.txt', run=False)
			transformed_results = obtain_transformed_detection(YOLO_path, load_existing_path+transf+"/bootstrap_transformed_list.txt", load_existing_path+transf+'/boot_transformed_'+model+'.txt', run=False)
		else:
			if only_read:
				orig_results = obtain_orig_detection(YOLO_path, "bootstrap_orig_list.txt", YOLO_path +'boot_orig_'+ model +'.txt', run=False)
				transformed_results = obtain_transformed_detection(YOLO_path, "bootstrap_transformed_list.txt", YOLO_path +'boot_transformed_'+model+'.txt', run=False)
			else:
				orig_results = obtain_orig_detection(YOLO_path, "bootstrap_orig_list.txt", YOLO_path +'boot_orig_'+model+'.txt', run=True)
				transformed_results = obtain_transformed_detection(YOLO_path, "bootstrap_transformed_list.txt", YOLO_path +'boot_transformed_'+model+'.txt', run=True)
		accuracy, f1 = print_ML_metrics(ground_truth, orig_results, transformed_results, labels)
		test_acc.append(accuracy)
		test_f1.append(f1)
		conf_abs, mu_abs, sigma_abs, conf_rel, mu_rel, sigma_rel = estimate_conf_int(ground_truth, orig_results, transformed_results, 1, labels)
		sat_abs.append(conf_abs)
		sat_rel.append(conf_rel)
		l_mu_abs.append(mu_abs)
		l_sigma_abs.append(sigma_abs)
		l_mu_rel.append(mu_rel)
		l_sigma_rel.append(sigma_rel)

	
		print('| '+ model  +' | ' + str(mu_abs) + ' | ' + str(sigma_abs) + ' | ' + str(conf_abs) + ' | ' + str(mu_rel) + ' | ' + str(sigma_rel) + ' | ' + str(conf_rel) + ' | ' + str(accuracy) + ' | ' + str(f1) + ' |')
		print('------------------------------------------------------------------------------------------------------')

	print("Correlations:")
	abs_acc, _ = pearsonr(sat_abs, test_acc)
	print("conf_acc and accuacy of test images: " + str(round(abs_acc, 3)))
	abs_f1, _ = pearsonr(sat_abs, test_f1)
	print("conf_acc and f1 of test images: " + str(round(abs_f1, 3)))
	rel_acc, _ = pearsonr(sat_rel, test_acc)
	print("conf_pred and accuracy of test images: " + str(round(rel_acc, 3)))
	rel_f1, _ = pearsonr(sat_rel, test_f1)
	print("conf_pred and f1 of test images: " + str(round(rel_f1, 3)))
	abs_rel, _ = pearsonr(sat_abs, sat_rel)
	print("conf_acc and conf_pred: " + str(round(abs_rel, 3)))
	