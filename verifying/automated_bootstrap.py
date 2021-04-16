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

# sample transformation parameter values with in the specified range, here chosen to be evenly distributed
# obtained through sampling images and checking the vd value in the requirements
intensity_shift_params = [-120, -100, -80, -60, -40 ,-20 , 20, 40, 60, 80, 100, 120]
gaussian_noise_params = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]
gamma_params = [0.9, 0.92, 0.94, 0.96, 0.98, 1.02, 1.04, 1.06, 1.08]

# transformations
def intensity_shift(img, degree, precision=0):  # state should contain values for the parameters
	#img = img.astype(float)
	return (img + np.around(degree, precision)).clip(min=0, max=255) 

def gaussian_noise(img, state, img_h, img_w, ch, precision=0): #state is (mean, sigma) sigma is standard deviation
	# currently gaussian with mean 0
	if ch > 1:
		gauss = np.random.normal(0,state,(img_h, img_w, ch))
	else:
		gauss = np.random.normal(0,state,(img_h, img_w))

	#print(gauss)
	#print(img)
	#gauss = gauss.reshape(self.img_h,self.img_w,self.ch)
	noisy = img + np.around(gauss, precision)
	return noisy.clip(min=0, max=255) 

def adjust_gamma(img, gamma, precision=0):
	return np.around(exposure.adjust_gamma(img, gamma), precision)

# first function: sample from all the classes (/Users/caroline/Desktop/REforML/HVS/experiment_code/orig_images)
# 100 batches of 50 images

def filter_image_classes(JPEG_files):
	mapping_file = open("MSCOCO_to_ImageNet_category_mapping.txt", "r").readlines()
	labels = re.findall('(n\d+)', ' '.join(mapping_file))
	#JPEG_files = open("inet.val.list", "r").readlines()
	#print(len(JPEG_files))
	def find_label(filename):
		label = re.findall('\.(n\d+)\.', filename)[0]
		if label in labels:
			return True
		else:
			return False
	JPEG_files = [f for f in JPEG_files if find_label(f) == True]
	#print(len(JPEG_files))
	#print(JPEG_files[0])

	#list_to_filter = []
	#print(labels)
	#while True:
	#	line = mapping_file.readline()
	#	if not line:
	#		break
		
	return JPEG_files


def gen_bootstrapping(num_batch, orig_path, gen_path, batch_size=50, transformation='intensity_shift'):
	if not os.path.exists(gen_path):
		os.makedirs(gen_path)
	#JPEG_files = [f for f in os.listdir(orig_path) if isfile(join(orig_path, f)) and f.endswith('JPEG')] #here it should be full path
	#list.sort(JPEG_files, key=find_num)
	JPEG_files = [y for x in os.walk(orig_path) for y in glob(os.path.join(x[0], '*.JPEG'))]
	JPEG_files = filter_image_classes(JPEG_files)
	#print(len(JPEG_files))
	#exit()
	# filter out other classes

	set_orig = []
	set_transformed = []
	for i in range(num_batch):
		print("batch " + str(i))
		batch = random.sample(JPEG_files, batch_size)
		set_orig += batch
		if not os.path.exists(gen_path + "batch_" +str(i) + '/'):
			os.makedirs(gen_path + "batch_" +str(i) + '/')
		store_path = gen_path + "batch_" +str(i) + '/'
		for f in batch:
			#image_name = f[:-5]
			image_name = re.findall('(ILSVRC2012_val_\d+)', f)[0]
			img = np.asarray(cv2.imread(f), dtype=np.float32)
			#print(img)
			# get image name
			#image_name = re.findall('\/(n\d+\_\d+).JPEG', f)#[0][0]
			#print(image_name)
			#exit()

			#gen_images(f, label, transformation, gen_path + "batch_" +str(i) + '/')
			if transformation == 'gaussian_noise':
				img_h, img_w, ch = img.shape
				param = random.choice(gaussian_noise_params)
				img2 = gaussian_noise(img, param, img_h, img_w, ch)
				cv2.imwrite(store_path+'_gaussian_' + str(param) + "_" + image_name + '.JPEG', img2)
				set_transformed.append(store_path+ '_gaussian_' + str(param) + "_" + image_name + '.JPEG')
			if transformation == 'intensity_shift':
				param = random.choice(intensity_shift_params)#for param in intensity_shift_params:
				img2 = intensity_shift(img, param)
				#print(store_path+ label+'_intensity_' + str(param) + "_" + image_name)
				cv2.imwrite(store_path+'_intensity_' + str(param) + "_" + image_name + '.JPEG', img2)
				set_transformed.append(store_path+ '_intensity_' + str(param) + "_" + image_name + '.JPEG')
			if transformation == 'gamma':
				param = random.choice(gamma_params)#for param in intensity_shift_params:
				img2 = adjust_gamma(img, param)
				cv2.imwrite(store_path+'_gamma_' + str(param) + "_" + image_name + '.JPEG', img2)
				set_transformed.append(store_path+ '_gamma_' + str(param) + "_" + image_name + '.JPEG')
	
	# write two txt files
	txt = open("bootstrap_orig_list.txt", "w")
	#print(all_transformed_files)
	#exit()
	for f in set_orig:
		txt.write(f + '\n')
	txt.close()

	txt = open("bootstrap_transformed_list.txt", "w")
	#print(all_transformed_files)
	#exit()
	for f in set_transformed:
		txt.write(f + '\n')
	txt.close()
	return set_orig, set_transformed

# obtain detection of the original images
def obtain_orig_detection(YOLO_path, list_orig_path, model, run=False):
	if run:
		os.popen("touch " + YOLO_path +"/boot_orig_"+model+ ".txt")
		pwd = os.popen("pwd").read()
		print(pwd)
		command = 'cd ' + YOLO_path + '; ./darknet classifier predict cfg/imagenet1k.data cfg/' + model +'.cfg '+ model + '.weights < ' +pwd.strip() + '/' + list_orig_path + ' > boot_orig_'+model+'.txt'
		print(command)
		cmd_result = os.popen(command).read()

	#exit()
	#results_f = open(YOLO_path + '/boot_orig_'+model+'.txt', "r")
	# if run = False, read existing
	if not run:
		results_f = open('bootstrap_results/boot_orig_'+model+'.txt', "r")
	else:
		results_f = open(YOLO_path +'boot_orig_'+model+'.txt', "r")
	file_name_f = open(list_orig_path, 'r')
	results = {}
	while True:
		line = results_f.readline()
		if not line:
			break
		if line.startswith('Enter Image Path:'):
			filename = re.findall('(ILSVRC2012_val_\d+)', file_name_f.readline())#[0]
			if filename != []:
				#print(line)
				matches = re.findall('([a-zA-Z\s]+$)', line)
				detection = matches[0].strip()
				if detection != '':
					results[filename[0]] = detection
	return results

# obtain detection of transformed images
def obtain_transformed_detection(YOLO_path, list_transformed_path, model, run=False):
	if run:
		os.popen("touch " + YOLO_path +"/boot_transformed_"+model+ ".txt")
		pwd = os.popen("pwd").read()
		print(pwd)
		command = 'cd ' + YOLO_path + '; ./darknet classifier predict cfg/imagenet1k.data cfg/' + model +'.cfg '+ model + '.weights < ' +pwd.strip() + '/' + list_transformed_path + ' > boot_transformed_'+model+'.txt'
		cmd_result = os.popen(command).read()

	#results_f = open(YOLO_path + '/boot_transformed_'+model+'.txt', "r")
	if not run:
		results_f = open('bootstrap_results/boot_transformed_'+model+'.txt', "r")
	else:
		results_f = open(YOLO_path +'boot_transformed_'+model+'.txt', "r")
	file_name_f = open(list_transformed_path, 'r')
	results = {}
	while True:
		line = results_f.readline()
		if not line:
			break
		if line.startswith('Enter Image Path:'):
			filename = file_name_f.readline()
			#print(line)
			matches = re.findall('([a-zA-Z\s]+$)', line)
			detection = matches[0].strip()
			if detection != '':
				results[filename.strip()] = detection
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

	ground_truth = {} # need to obtain from synset_words.txt
	image_list_f = open(image_list, "r")
	image_names_set = []
	while True:
	# Get next line from file
		line = image_list_f.readline()
		if not line:
			break
		# this may change based on the file we are running
		matches = re.findall('(ILSVRC2012\_val\_\d+)\.(n\d+)', line) #validation
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
			#print(line)
			matches = re.findall('([a-zA-Z\s]+$)', line)
			detection = matches[0].strip()
			if detection != '':
				results[image_names_set[cur_id]] = detection
				#print("detection " + str(cur_id) +" : " + image_names_set[cur_id] + " -> " + detection)
				cur_id += 1
	#for image in ground_truth.keys():
	#	print(image + " ground truth: " + str(ground_truth[image]) + "; detection: " + results[image])
	return ground_truth, results

def calculate_confidence(acc_list, base_acc, req_acc):
	# fitting a normal distribution
	_, bins, _ = plt.hist(acc_list, 20, alpha=0.5, density=True)
	mu, sigma = scipy.stats.norm.fit(acc_list)
	best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
	print("Estimated mean from bootstrapping: " + str(mu))
	print("Estimated sigma from bootstrapping: " + str(sigma))
	# calculate the requirement
	#requirement_acc = base_acc - x_squared_inverse(IQA_value) 
	#print("requirement is " + str(requirement_acc))
	#print(bins)
	#plt.plot(bins, best_fit_line, label="Normal Distribution")
	#plt.axvline(x=req_acc, label="Accuracy from the Requirement")
	#plt.legend()
	#plt.show()
	
	distribution = scipy.stats.norm(mu, sigma)
	#if base_acc < mu:
	#	result = distribution.cdf(base_acc)
	#else:
	result = 1- distribution.cdf(req_acc)
	#print(distribution.cdf(requirement_acc))
	print('confidence of satisfication:' + str(result))
	if result >= 0.5:
		print("requirement SATISFIED")
	else:
		print("requirement NOT SATISFIED")
	return result

def estimate_conf_int(ground_truth, orig_results, transformed_result, a): #rel or abs
	car_labels = ['beach wagon', 'station wagon', 'wagon', 'estate car', 'beach waggon', 'station waggon', 'waggon', 'convertible', 'sports car', 'sport car', 'flamingo']

	batch_results_abs = {}
	batch_results_rel = {}
	#sort batch results
	for trans_img in transformed_result.keys():
		#print(trans_img)
		matches = re.findall('(batch\_\d+).*(ILSVRC2012_val_\d+)', trans_img)
		orig_img_name = matches[0][1]
		batch = matches[0][0]
		#print(orig_img_name)
		#print(orig_img_name, trans_img)
		#print(transformed_result[trans_img], ground_truth[orig_img_name])
		#continue
		#print(orig_img_name, batch)
		if batch not in batch_results_abs.keys():
			batch_results_abs[batch] = {}
		if batch not in batch_results_rel.keys():
			batch_results_rel[batch] = {}
		#print(ground_truth[orig_img_name][0])
		if ground_truth[orig_img_name][0] in car_labels:
			batch_results_abs[batch][orig_img_name] = transformed_result[trans_img]
		if transformed_result[trans_img] in car_labels:
			batch_results_rel[batch][orig_img_name] = transformed_result[trans_img]
	#print(batch_results)

	# obtain abs
	batch_accuracies = []
	for batch in batch_results_abs.keys():
		#print(sum([1 for x in batch_results[batch].keys() if batch_results[batch][x] in ground_truth[x]]))
		batch_accuracies.append(sum([1 for x in batch_results_abs[batch].keys() if batch_results_abs[batch][x] in ground_truth[x]])/len(batch_results_abs[batch].keys()))

	batch_preserved = []
	for batch in batch_results_rel.keys():
		batch_preserved.append(sum([1 for x in batch_results_rel[batch].keys() if (batch_results_rel[batch][x] in car_labels) == (orig_results[x] in car_labels)]  )/len(batch_results_rel[batch].keys()))

	#print("ABS batches")
	#print(batch_accuracies)
	#print("REL batches")
	#print(batch_preserved)

	base_acc = sum([1 for x in orig_results.keys() if orig_results[x] in ground_truth[x]])/len(orig_results.keys())
	print("--------------------------------------------")
	print("1. Verifying Absolute Requirement: ")
	conf_abs = calculate_confidence(batch_accuracies, base_acc, base_acc) #abs
	print("--------------------------------------------")
	print("2. Verifying Relative Requirement:")
	conf_rel = calculate_confidence(batch_preserved, base_acc, a) #rel
	print("--------------------------------------------")

	return conf_abs, conf_rel

def print_metrics(ground_truth, detections):
	car_labels = ['beach wagon', 'station wagon', 'wagon', 'estate car', 'beach waggon', 'station waggon', 'waggon', 'convertible', 'sports car', 'sport car']
	true_pos = 0
	true_neg = 0
	false_pos = 0
	false_neg = 0
	for image in detections.keys():

		if ground_truth[image][0] in car_labels: 
			if detections[image] in car_labels: # true positive
				true_pos += 1
			else: 	# false negative
				false_neg += 1
		else: #negatives
			if detections[image] in car_labels: # false positive
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
	car_labels = ['beach wagon', 'station wagon', 'wagon', 'estate car', 'beach waggon', 'station waggon', 'waggon', 'convertible', 'sports car', 'sport car']
	true_pos = 0
	true_neg = 0
	false_pos = 0
	false_neg = 0
	for image in batch_result.keys():
		image_name = re.findall('(ILSVRC2012_val_\d+)', image)[0]
		gt = ground_truth[image_name][0]
		if gt in car_labels: 
			if batch_result[image] in car_labels: # true positive
				true_pos += 1
			else: 	# false negative
				false_neg += 1
		else: #negatives
			if batch_result[image] in car_labels: # false positive
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

def usage():
	#usage = ""
	parser=argparse.ArgumentParser(
    description='''My Description. And what a lovely description it is. ''',
    epilog="""All is well that ends well.""")
	#parser.add_argument('--foo', type=int，help='FOO!')
	parser.add_argument('-d', '--Darknet_path', type=str, help='path to Darknet directory')
	parser.add_argument('-r', '--read_existing', type=str, help='read from exisitng run')
	parser.add_argument('-i', '--image_path', type=str, help='path to ILSVRC2012 validation images')
	parser.add_argument('-n', '--number_of_batches', type=int, help='the number of batches for bootstrapping, must be integer')
	parser.add_argument('-s', '--batch_size', type=int, help='the number of images per batch for bootstrapping, must be integer')
	#parser.add_argument('bar', nargs='*', default=[1, 2, 3], help='BAR!')
	args=parser.parse_args()
	#return usage
if __name__ == "__main__":
	try:
		opts, args = getopt.getopt(sys.argv[1:], "hd:ri:n:s:", ["help", "Darknet_path=", "read_existing", "image_path", "number_of_batches", "batch_size"])
	except getopt.GetoptError as err:
		# print help information and exit:
		print(err)  # will print something like "option -a not recognized"
		#usage()
		sys.exit(2)
	YOLO_path = None
	read_exist = False
	image_path = None
	number_of_batches = 200
	batch_size = 500
	for o, a in opts:
		if o in ("-r", "--read_existing"):#o == "-v": 
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
		#else:
			#assert False, "unhandled option"
		#	print("invalid argument\n")
		#	usage()
		#	sys.exit()

	if not read_exist and not image_path:
		print("missing path to validation images")
		sys.exit()
	if not read_exist:
		gen_bootstrapping(number_of_batches, image_path, 'bootstrap_samples/', batch_size=batch_size)
	#gen_bootstrapping(1, '/Users/caroline/Desktop/REforML/HVS/scripts/val_img_classes/', 'bootstrap_samples/')
	#ground_truth, detections = read_result("/u/boyue/darknet/validation_darknet19.txt", '/u/boyue/Desktop/inet.val.list')
	models = ['darknet19', 'darknet53_448', 'alexnet', 'resnext50', 'vgg-16', 'resnet50']
	sat_abs = []
	sat_rel = []
	val_acc = []
	val_f1 = []
	trans_acc = []
	trans_f1 = []
	for model in models:
		print("###################" + model + "###################")
		time_start = time.process_time()
		ground_truth, detections = read_result("bootstrap_results/validation_"+ model +".txt", 'bootstrap_results/inet.val.list')
		print("--------------------------------------------")
		print("i. Precision and f1 on ILSVRC2012 validation set:")
		accuracy, f1 = print_metrics(ground_truth, detections)
		val_acc.append(accuracy)
		val_f1.append(f1)
		if read_exist:
			orig_results = obtain_orig_detection(YOLO_path, "bootstrap_results/bootstrap_orig_list.txt", model, run=False)
			transformed_results = obtain_transformed_detection(YOLO_path, "bootstrap_results/bootstrap_transformed_list.txt", model, run=False)
		else:
			orig_results = obtain_orig_detection(YOLO_path, "bootstrap_orig_list.txt", model, run=True)
			transformed_results = obtain_transformed_detection(YOLO_path, "bootstrap_transformed_list.txt", model, run=True)
		print("ii. Precision and f1 on sampled transformed images from bootstrap:")
		accuracy, f1 = print_metrics_batch_result(ground_truth, transformed_results)
		trans_acc.append(accuracy)
		trans_f1.append(f1)
		conf_abs, conf_rel = estimate_conf_int(ground_truth, orig_results, transformed_results, 0.95)
		time_elapsed = (time.process_time() - time_start)
		sat_abs.append(conf_abs)
		sat_rel.append(conf_rel)
		print("3. Print runtime and memory useage")
		print(resource.getrusage(resource.RUSAGE_SELF))
	print("##########################################")
	
	print('********************correlation results**********************')
	print('-----------------------------------------------------------------------------------------')
	print('|                 Satisfying Absolute VS standard ML reliability metrics                |')
	print('-----------------------------------------------------------------------------------------')
	print('|           | metrics for ILSVRC2012 validation | metrics for sampled transformed images|')
	print('-----------------------------------------------------------------------------------------')
	print('| corr type |    accuracy     |        f1       |      accuracy       |        f1       |')
	print('-----------------------------------------------------------------------------------------')
	acc_corr, _ = pearsonr(sat_abs, val_acc)
	f1_corr, _ = pearsonr(sat_abs, val_f1)
	t_acc_corr, _ = pearsonr(sat_abs, trans_acc)
	t_f1_corr, _ = pearsonr(sat_abs, trans_f1)
	print('| Pearson   |      ' + str(round(acc_corr, 3)) + '      |      '  + str(round(f1_corr, 3)) + '      |       ' + str(round(t_acc_corr, 3)) + '         |     ' + str(round(t_f1_corr, 3)) + '       |')
	print('-----------------------------------------------------------------------------------------')
	acc_corr, _ = spearmanr(sat_abs, val_acc)
	f1_corr, _ = spearmanr(sat_abs, val_f1)
	t_acc_corr, _ = spearmanr(sat_abs, trans_acc)
	t_f1_corr, _ = spearmanr(sat_abs, trans_f1) 
	print('| Spearman  |       ' + str(acc_corr) + '       |       ' + str(f1_corr) + '       |        ' + str(t_acc_corr) + '          |      ' + str(t_f1_corr) + '        |')
	print('-----------------------------------------------------------------------------------------')
	print('\n')
	print('-----------------------------------------------------------------------------------------')
	print('|                 Satisfying Relative VS standard ML reliability metrics                |')
	print('-----------------------------------------------------------------------------------------')
	print('|           | metrics for ILSVRC2012 validation | metrics for sampled transformed images|')
	print('-----------------------------------------------------------------------------------------')
	print('| corr type |    accuracy     |        f1       |      accuracy       |        f1       |')
	print('-----------------------------------------------------------------------------------------')
	acc_corr, _ = pearsonr(sat_rel, val_acc)
	f1_corr, _ = pearsonr(sat_rel, val_f1)
	t_acc_corr, _ = pearsonr(sat_rel, trans_acc)
	t_f1_corr, _ = pearsonr(sat_rel, trans_f1)
	print('| Pearson   |      ' + str(round(acc_corr, 3)) + '      |      '  + str(round(f1_corr, 3)) + '      |       ' + str(round(t_acc_corr, 3)) + '         |     ' + str(round(t_f1_corr, 3)) + '       |')
	print('-----------------------------------------------------------------------------------------')
	acc_corr, _ = spearmanr(sat_rel, val_acc)
	f1_corr, _ = spearmanr(sat_rel, val_f1)
	t_acc_corr, _ = spearmanr(sat_rel, trans_acc)
	t_f1_corr, _ = spearmanr(sat_rel, trans_f1) 
	print('| Spearman  |      ' + str(round(acc_corr, 3)) + '      |      '  + str(round(f1_corr, 3)) + '      |       ' + str(round(t_acc_corr, 3)) + '         |     ' + str(round(t_f1_corr, 3)) + '       |')
	print('-----------------------------------------------------------------------------------------')

	print("Correlation between the satisfaction of two requirements:")
	corr, _ = pearsonr(sat_rel, sat_abs)
	print('pearson correlation: %.3f' % corr)
	corr, _ = spearmanr(sat_rel, sat_abs)
	print('spearman correlation: %.3f' % corr)
