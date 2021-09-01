import csv
import os
import numpy
import random
import pickle
import csv
import re
import pandas as pd
import matplotlib.pyplot as plt
import math
from collections import Counter
from scipy import stats
import numpy as np

IQA_interval = 0.01
display = True
for_RQ1 = False
DATASET = 'imagenet'
threshold_size = 10

if DATASET == 'cifar10':
	cifar10_exp_results = 'cifar10_experiment_results.csv'
	classification_results = []
	with open(cifar10_exp_results, mode='r') as exp_results:
		csv_reader = csv.DictReader(exp_results)
		for row in csv_reader:
			img = row['image_link']
			if 'IMG_ORIGINAL' in img:
				filename = re.findall(r"(([^\/]*)\/([^\/]*).png)$", img)[0]
				identifier = filename[2]
				classification_results.append((float(row['IQA']), row['human_label'], row['ground_truth'], filename, row['transformation'], identifier))
				
			else:
				filename_match = re.findall(r"(([^\/]*)\/([^\/]*).png)$", img)
				filename = filename_match[0][0]
				transf = filename_match[0][1]
				identifier = filename_match[0][2]
				
				classification_results.append((float(row['IQA']), row['human_label'], row['ground_truth'], filename, row['transformation'], identifier))
else:
	imagenet_exp_results = 'imagenet_experiment_results.csv'
	classification_results = []
	with open(imagenet_exp_results, mode='r') as exp_results:
		csv_reader = csv.DictReader(exp_results)
		for row in csv_reader:
			img = row['image_link']
			filename_match = re.findall(r"\/([^\/]*.JPEG\.([^\/]*)_\d+.png)", img) # first check if it's original image
			
			if len(filename_match) == 0:
				filename = re.findall(r"\/([^\/]*.JPEG)$", img)[0]
				identifier_match = re.findall(r"((n\d+)_\d+)",img)[0]
				identifier = identifier_match[0]
				classification_results.append((float(row['IQA']), row['human_label'], row['ground_truth'], filename, row['transformation'], identifier))
				
			else:
				filename = filename_match[0][0]
				transf = filename_match[0][1]
				identifier_match = re.findall(r"((n\d+)_\d+)", filename)
				identifier = identifier_match[0][0]
				
				classification_results.append((float(row['IQA']), row['human_label'], row['ground_truth'], filename, row['transformation'], identifier))
				#classes = set([tp[2] for tp in classification_results])
transformations = set([tp[4] for tp in classification_results if tp[4] != 'original'])
############## correctness_preservation #############
correctness_preservation = {}
for t in transformations:
	results_transformation = [r for r in classification_results if r[-2] == t]
	accuracy_list = []
	i = 0
	while i <= 1:
		i = round(i, 2)
		interval_result = [r for r in results_transformation if (1-r[0]) <= i and (1-r[0]) > i-IQA_interval]
		if for_RQ1:
			interval_result = random.sample(interval_result, int(len(interval_result) * 0.6))
		if len(interval_result) > threshold_size:
			acc = len([r for r in interval_result if r[1] == r[2]])/len(interval_result)
			accuracy_list.append((i, acc, len(interval_result), acc, len([r for r in interval_result if r[1] == r[2]])))
		i += IQA_interval
		i = round(i, 2)
	correctness_preservation[t] = accuracy_list

	if display:
		plt.title('Mapping for each transformation ')
		plt.ylim([0, 1])
		plt.plot([x[0] for x in accuracy_list if x[2] >= threshold_size], [x[3] for x in accuracy_list if x[2] >= threshold_size], '.-', label = t)
		for info in accuracy_list:
			if info[2] >= threshold_size:
				plt.annotate((info[2],info[0]), (info[0], info[3]))
	if display:
		plt.ylabel('Accuracy')
		plt.xlabel("visual_degrade")
		plt.legend()
		plt.show()

for t in transformations:
	f = open( DATASET+'_c_p_'+t+'.csv', 'w')
	writer = csv.writer(f)
	header = ['IQA', 'Accuracy', 'Count', 'Num_Accs']
	writer.writerow(header)
	for entry in correctness_preservation[t]:
		row = [str(entry[0]),str(entry[1]),str(entry[2]),str(entry[-1])]
		writer.writerow(row)
	f.close()

############## relative #############
relative_results = []

for single_result in classification_results:
	if single_result[-2] != 'original':
		list_original = [tp[1] for tp in classification_results if tp[-1] == single_result[-1] and tp[-2] == 'original']#[tp[1] for tp in classification_results if tp[3] == single_result[3] and tp[-2] == 'original']
		if len(list_original) == 0:
			continue
		c = Counter(list_original)
		value, count = c.most_common()[0]
		relative_results.append((single_result[0], single_result[1], value, single_result[-3], single_result[-2]))			

prediction_preservation = {}
for t in transformations:
	results_transformation = [r for r in relative_results if r[-1] == t] 
	accuracy_list = []
	i = 0
	while i <= 1:
		i = round(i, 2)
		interval_result = [r for r in results_transformation if (1-r[0]) <= i and (1-r[0]) > i-IQA_interval]
		if for_RQ1:
			interval_result = random.sample(interval_result, int(len(interval_result) * 0.9))
		if len(interval_result) > threshold_size:
			acc = len([r for r in interval_result if r[1] == r[2]])/len(interval_result)
			accuracy_list.append((i, acc, len(interval_result), acc, len([r for r in interval_result if r[1] == r[2]])))
		i += IQA_interval
		i = round(i, 2)
	prediction_preservation[t] = accuracy_list

	if display:
		plt.title('Mapping for each transformation ')
		plt.ylim([0, 1])
		plt.plot([x[0] for x in accuracy_list if x[2] >= threshold_size], [x[3] for x in accuracy_list if x[2] >= threshold_size], '.-', label = t)
		for info in accuracy_list:
			if info[2] >= threshold_size:
				plt.annotate((info[2], info[0]), (info[0], info[3]))
	if display:
		plt.ylabel('Percentage preserved')
		plt.xlabel("visual_degrade")
		plt.legend()
		plt.show()

for t in transformations:
	f = open( DATASET+'_p_p_'+t+'.csv', 'w')
	writer = csv.writer(f)
	header = ['IQA', 'Preserved', 'Count', 'Num_Pre']
	writer.writerow(header)
	for entry in prediction_preservation[t]:
		row = [str(entry[0]),str(entry[1]),str(entry[2]),str(entry[-1])]
		writer.writerow(row)
	f.close()
	