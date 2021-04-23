import tarfile
import os
import pickle
from os import listdir
from os.path import isfile, join
import csv
import re
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize, least_squares, curve_fit
from scipy import stats
import numpy as np
import getopt
import argparse
import sys

csv_files_dir = './csv_files/'
new_csv_file_dir = './new_csv/'

def write_new_csv():
	# load the name to IQAs to a big dictionary
	#dict2.update(dict1)
	
	filename_to_IQA =pickle.load(open('all_filename_to_IQA.pkl', 'rb'))
	new_pkl_files = [f for f in listdir('.') if isfile(f) and 'pkl' in f]

	for pkl in new_pkl_files:
		cur_dict = pickle.load(open(pkl, "rb" ))
		filename_to_IQA.update(cur_dict)

	pickle.dump(filename_to_IQA, open('all_filename_to_IQA.pkl', 'wb'))

	# loop over old csv files to find the ones in the dictionary
	csv_files = [f for f in listdir(csv_files_dir) if isfile(join(csv_files_dir, f)) and 'csv' in f]
	for csv_name in csv_files:
		# start a corresponding document for the IQA values
		f = open(new_csv_file_dir + "new_"+csv_name, "w")
		f.write("subj,session,trial,rt,object_response,category,condition,imagename,IQA\n")
		with open(csv_files_dir + csv_name, newline='') as csv_file:
			raw_data = csv.reader(csv_file)
			for row in raw_data:
				filename = row[-1]
				parameter = row[6]
				if 'contrast' in csv_name and parameter == 'c100':
					separator = ','
					new_row = separator.join(row) + ',1\n'
					f.write(new_row)
				elif 'noise' in csv_name and parameter == '0.00':
					separator = ','
					new_row = separator.join(row) + ',1\n'
					f.write(new_row)
				elif 'rotation' in csv_name and parameter == '0':
					separator = ','
					new_row = separator.join(row) + ',1\n'
					f.write(new_row)
				elif 'highpass' in csv_name and parameter == 'inf':
					separator = ','
					new_row = separator.join(row) + ',1\n'
					f.write(new_row)
				elif 'lowpass' in csv_name and parameter == '0':
					separator = ','
					new_row = separator.join(row) + ',1\n'
					f.write(new_row)
				elif 'phase-scrambling' in csv_name and parameter == '0':
					separator = ','
					new_row = separator.join(row) + ',1\n'
					f.write(new_row)
				elif filename in filename_to_IQA.keys():
					separator = ','
					new_row = separator.join(row) + "," + str(filename_to_IQA[filename]) + '\n'
					f.write(new_row)

def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum

def find_labels_preserved():
	classes = ['dog', 'bird', 'car', 'truck', 'bottle', 'cat', 'boat', 'bear', 'chair', 'oven', 'clock', 'airplane', 'bicycle', 'elephant', 'keyboard', 'knife']
	transformations = ['contrast', 'noise', 'rotation', 'highpass', 'lowpass', 'phase-scrambling']
	csv_files = [f for f in listdir(new_csv_file_dir) if isfile(join(new_csv_file_dir, f)) and 'csv' in f]
	all_results = {}
	all_base_results = {}
	for f in csv_files:
		with open(new_csv_file_dir + f, newline='') as csv_file:
			raw_data = csv.reader(csv_file)
			for row in raw_data:
				if row[-1] != 'IQA':
					IQA_value = float(row[-1])
					subject = row[0]
					ground_truth = row[5]
					detection = row[4]
					parameter = row[6]
					img_name = re.findall('(n\d+_\d+)\..+', row[-2])[0]
					#print(img_name)
					if 'contrast' in f:
						
						if parameter == 'c100':
							#print(row)
							if img_name not in all_base_results.keys():
								all_base_results[img_name] = {}
							all_base_results[img_name][subject] = (ground_truth, detection)
							#print(img_name, subject, all_base_results[img_name][subject])
						else:
							all_results[(row[-2], subject)] = (ground_truth, detection, 1-IQA_value, "contrast")
					if 'noise' in f:
						
						if parameter == '0.00':
							#print(row)
							if img_name not in all_base_results.keys():
								all_base_results[img_name] = {}
							all_base_results[img_name][subject] = (ground_truth, detection)
							#print(img_name, subject, all_base_results[img_name][subject])
						else:
							all_results[(row[-2], subject)] = (ground_truth, detection, 1-IQA_value, "noise")

					if 'rotation' in f:
						
						if parameter == '0':
							#print(row)
							if img_name not in all_base_results.keys():
								all_base_results[img_name] = {}
							all_base_results[img_name][subject] = (ground_truth, detection)
							#print(img_name, subject, all_base_results[img_name][subject])
						else:
							all_results[(row[-2], subject)] = (ground_truth, detection, 1-IQA_value, "rotation")

					if 'highpass' in f:
						
						if parameter == 'inf': 
							#print(row)
							if img_name not in all_base_results.keys():
								all_base_results[img_name] = {}
							all_base_results[img_name][subject] = (ground_truth, detection)
							#print(img_name, subject, all_base_results[img_name][subject])
						else:
							all_results[(row[-2], subject)] = (ground_truth, detection, 1-IQA_value, "highpass")
					if 'lowpass' in f:
						
						if parameter == '0':
							#print(row)
							if img_name not in all_base_results.keys():
								all_base_results[img_name] = {}
							all_base_results[img_name][subject] = (ground_truth, detection)
							#print(img_name, subject, all_base_results[img_name][subject])
						else:
							all_results[(row[-2], subject)] = (ground_truth, detection, 1-IQA_value, "lowpass")
					if 'eidolon' in f:
						if parameter.split('-')[0] == '1':
							if img_name not in all_base_results.keys():
								all_base_results[img_name] = {}
							all_base_results[img_name][subject] = (ground_truth, detection)
						else:
							all_results[(row[-2], subject)] = (ground_truth, detection, 1-IQA_value, "eidolon")
					if 'phase-scrambling' in f:
						#print('found ps')
						if parameter == '0':
							if img_name not in all_base_results.keys():
								all_base_results[img_name] = {}
							all_base_results[img_name][subject] = (ground_truth, detection)
						else:
							all_results[(row[-2], subject)] = (ground_truth, detection, 1-IQA_value, "phase-scrambling")
		csv_file.close()

	# process checking whether the detection has been preserved
	wrong_right_count = 0
	wrong_wrong_count = 0
	wrong_wrong_same = 0
	results_tuples = []
	for (full_img_name, sub) in all_results.keys():
		a, b, c, d = all_results[(full_img_name, sub)]
		img_name = re.findall('(n\d+_\d+)\..+', full_img_name)[0]
		if img_name in all_base_results.keys():
			orig_detection = find_majority([t[1] for t in all_base_results[img_name].values()])[0]
			results_tuples.append((a, b, c, d, orig_detection)) #ground truth, detection, IQA, transformation, orig_detection
			#TODO: how often does wrong -> right happen? #254/8908
			# how often does wrong still wrong but the same happen? #408/8908
			if (orig_detection != a) and (b == a):
				wrong_right_count += 1
			if (orig_detection != a) and (a != b) and (orig_detection != b):
				wrong_wrong_count += 1
			if (orig_detection != a) and (a != b) and (orig_detection == b): 
				wrong_wrong_same += 1

	#print(str(wrong_right_count) + '/' + str(len(results_tuples)))
	#print(str(wrong_wrong_count) + '/' + str(len(results_tuples)))
	#print(str(wrong_wrong_same) + '/' + str(len(results_tuples)))
	return all_base_results, results_tuples

def all_class_diff_trans(results_tuples, object_class, interval, display=True):
	classes = ['dog', 'bird', 'car', 'truck', 'bottle', 'cat', 'boat', 'bear', 'chair', 'oven', 'clock', 'airplane', 'bicycle', 'elephant', 'keyboard', 'knife']

	transformations = ['contrast', 'noise', 'highpass', 'lowpass', "rotation", 'phase-scrambling']

	result_transformation = {}
	result_curves = {}

	# first filter all the results with this class
	results_all_transformations = [t for t in results_tuples]#! if t[0] == c]
	result_transformation = {}
	#then look at all the transformations
	for t in transformations:
		results_transformation = [r for r in results_all_transformations if r[3] == t]

		accuracy_list = []

		i = 0
		while i <= 1:
			i = round(i, 2)
			interval_result = [r for r in results_transformation if r[2] <= i and r[2] > i-interval]
			if len(interval_result) >=10:
				acc = len([r for r in interval_result if (r[1]== object_class)==(r[-1]==object_class)])/len(interval_result)
				accuracy_list.append((i, acc, len(interval_result), acc, len([r for r in interval_result if (r[1]== object_class)==(r[-1]==object_class)])))
			i += interval
			i = round(i, 2)
		result_transformation[t] = accuracy_list#![c][t] = accuracy_list
		#each_transformation[t] = accuracy_list
		if display:
			plt.title('Mapping for all classes ' )
			plt.plot([x[0] for x in accuracy_list if x[2] >= 10], [x[3] for x in accuracy_list if x[2] >= 10], '.-', label = t)
			for info in accuracy_list:
				if info[2] >= 10:
					plt.annotate(info[2], (info[0], info[3]))
	if display:
		plt.legend()
		plt.show()		
	return result_transformation

def find_decrease_in_percentage(interval, display=True):
	# mean, max/min accuracy for the IQA value intervals
	classes = ['dog', 'bird', 'car', 'truck', 'bottle', 'cat', 'boat', 'bear', 'chair', 'oven', 'clock', 'airplane', 'bicycle', 'elephant', 'keyboard', 'knife']
	#classes = ['car']
	transformations = ['contrast', 'noise', 'rotation', 'highpass', 'lowpass', 'phase-scrambling']
	#transformations = ['contrast', 'noise', 'highpass', 'lowpass', 'eidolon', 'phase-scrambling']
	csv_files = [f for f in listdir(new_csv_file_dir) if isfile(join(new_csv_file_dir, f)) and 'csv' in f]
	all_results = []
	all_base_results = []
	for f in csv_files:
		with open(new_csv_file_dir + f, newline='') as csv_file:
			raw_data = csv.reader(csv_file)
			for row in raw_data:
				if row[-1] != 'IQA':
					IQA_value = float(row[-1])
					detection = (row[4] == row[5])
					parameter = row[6]
					if 'contrast' in f:
						all_results.append((1-IQA_value, detection, row[0], row[5], 'contrast'))
						if parameter == 'c100' and IQA_value==1:
							all_base_results.append((1- IQA_value, detection, row[0], row[5], 'contrast'))
					if 'noise' in f:
						all_results.append((1- IQA_value, detection, row[0], row[5], 'noise'))
						if parameter == '0.00' and IQA_value==1:
							all_base_results.append((1-IQA_value, detection, row[0], row[5], 'noise'))
					if 'rotation' in f:
						all_results.append((1-IQA_value, detection, row[0], row[5], 'rotation'))
						if parameter == '0' and IQA_value==1:
							all_base_results.append((1-IQA_value, detection, row[0], row[5], 'rotation'))
					if 'highpass' in f:
						all_results.append((1-IQA_value, detection, row[0], row[5], 'highpass'))
						if parameter == 'inf' and IQA_value==1:
							all_base_results.append((1-IQA_value, detection, row[0], row[5], 'highpass'))
					if 'lowpass' in f:
						all_results.append((1-IQA_value, detection, row[0], row[5], 'lowpass'))
						if parameter == '0' and IQA_value==1:
							all_base_results.append((1-IQA_value, detection, row[0], row[5], 'lowpass'))
					if 'eidolon' in f:
						all_results.append((1-IQA_value, detection, row[0], row[5], 'eidolon'))
						if parameter.split('-')[0] == '1':
							all_base_results.append((0, detection, row[0], row[5], 'eidolon'))
					if 'phase-scrambling' in f:
						all_results.append((1-IQA_value, detection, row[0], row[5], 'phase-scrambling'))
						if parameter == '0':
							all_base_results.append((0, detection, row[0], row[5], 'phase-scrambling'))
					#	ps_result.append((IQA_value, detection, row[0], row[5]))
	# for different classes
	final_all_results = {}

	for c in classes:
		# first filter all the results with this class
		results_all_transformations = [t for t in all_results if t[3] == c]
		base_results_all_transformations = [t for t in all_base_results if t[3] == c]
		final_all_results[c] = {}
		#then look at all the transformations
		for t in transformations:
			results_transformation = [r for r in results_all_transformations if r[-1] == t]
			base_results_transformation = [r for r in base_results_all_transformations if r[-1] == t]
			if len(base_results_transformation) > 0:
				base_accuracy_transformation = len([t for t in base_results_transformation if t[1]])/len(base_results_transformation)
			accuracy_list = []

			i = 0
			while i <= 1:
				i = round(i, 2)
				interval_result = [r for r in results_transformation if r[0] <= i and r[0] > i-interval]
				if len(interval_result) > 0:
					acc = len([r for r in interval_result if r[1]])/len(interval_result)
					accuracy_list.append((i, acc, len(interval_result), acc, len([r for r in interval_result if r[1]])))

				i += interval
				i = round(i, 2)
			final_all_results[c][t] = accuracy_list
			if display:
				plt.title('Mapping for class ' + c)
				plt.plot([x[0] for x in accuracy_list if x[2] >= 10], [x[3] for x in accuracy_list if x[2] >= 10], '.-', label = t)
				for info in accuracy_list:
					if info[2] >= 10:
						plt.annotate(info[2], (info[0], info[3]))
		if display:
			plt.ylabel('Accuracy')
			plt.xlabel("visual_degrade")
			plt.legend()
			plt.show()

	return final_all_results, all_results, all_base_results

def usage():
	#usage = ""
	parser=argparse.ArgumentParser(
	description='''This is the script for our subprocess IV Verifying MLC against requirement ''')
	parser.add_argument('-c', '--class', type=str, help='the class of object to estimate the parameter, default is car')
	
	args=parser.parse_args()
if __name__ == "__main__":
	list_labels = ['airplane', 'bicycle', 'boat', 'car', 'chair', 'dog', 'keyboard', 'oven', 'bear', 'bird', 'bottle', 'cat', 'clock', 'elephant', 'knife', 'truck']
	transformations = ['contrast', 'noise', 'highpass', 'lowpass',  'phase-scrambling']

	try:
		opts, args = getopt.getopt(sys.argv[1:], "hc:", ["help", "class="])
	except getopt.GetoptError as err:
		# print help information and exit:
		print(err)  
		usage()
		sys.exit(2)

	object_class = 'car'

	for o,a in opts:
		if o in ("-h", "--help"):
			usage()
			sys.exit()
			only_read = True
		elif o in ("-c", "--class"):
			object_class = a.lower().strip()
			if object_class not in list_labels:
				print("invalid class, must be from one of ['airplane', 'bicycle', 'boat', 'car', 'chair', 'dog', 'keyboard', 'oven', 'bear', 'bird', 'bottle', 'cat', 'clock', 'elephant', 'knife', 'truck']")
				usage()
				sys.exit()
		else:
			assert False, "unhandled option"
			print("invalid argument\n")
			usage()
			sys.exit()
	

	# -load, -dont_show
	write_new_csv()
	
	# TODO: need to change the accuracy measure to car not car
	# for relative
	all_base_result, results_tuples = find_labels_preserved()
	result_transformation = all_class_diff_trans(results_tuples, 'car', 0.01, False)
	
	t_diff_trans = []
	a_diff_trans = []
	print("For prediction preservation:")
	for t in transformations:
		cur_acc_array = result_transformation[t]
		a = cur_acc_array[0][1]
		a_diff_trans.append(a)
		vd_score_to_p_value = []
		for entry in cur_acc_array:
			p_value = stats.binom_test(entry[-1], n=entry[2], p=a)
			vd_score_to_p_value.append(p_value)
		i = len(vd_score_to_p_value)-1
		while i > 0:
			if (vd_score_to_p_value[i] > 0.05):
				break
			i = i -1
		t_diff_trans.append(cur_acc_array[i-1][0])
		print("threshold t for " + t + ": " + str(cur_acc_array[i-1][0]))
		print("human percentage a for " + t + ": " + str(a))
	
	if all(elem == t_diff_trans[0] for elem in t_diff_trans):
		print("generalized t = " + str(t_diff_trans[0]))
	else:
		t = stats.t.interval(0.95, len(t_diff_trans)-1, loc=np.mean(t_diff_trans), scale=stats.sem(t_diff_trans))
		print("generalized t = " + str(t[0]))

	if all(elem == a_diff_trans[0] for elem in a_diff_trans):
		print("generalized a = "+ str(a_diff_trans[0]))
	else:
		a = stats.t.interval(0.95, len(a_diff_trans)-1, loc=np.mean(a_diff_trans), scale=stats.sem(a_diff_trans))
		print("generalized a = "+ str(a[0]))

	# for absolute
	final_all_results, all_results, all_base_results = find_decrease_in_percentage(0.01, False)
	final_object_results = final_all_results[object_class]
	base_results_all_transformations = [t for t in all_base_results if t[3] == object_class]
	t_diff_trans = []
	print("For accuracy preservation:")
	for t in transformations:
		cur_acc_array = final_object_results[t]
		base_results_transformation = [r for r in base_results_all_transformations if r[-1] == t]
		if len(base_results_transformation) > 0:
			base_acc = len([t for t in base_results_transformation if t[1]])/len(base_results_transformation)
		vd_score_to_p_value = []
		for entry in cur_acc_array:
			p_value = stats.binom_test(entry[-1], n=entry[2], p=base_acc)
			vd_score_to_p_value.append(p_value)
		i = len(vd_score_to_p_value)-1
		while i > 0:
			if (vd_score_to_p_value[i] > 0.05):
				break
			i = i -1

		t_diff_trans.append(cur_acc_array[i-1][0])
		print("threshold t for " + t + ": " + str(cur_acc_array[i-1][0]))

	if all(elem == t_diff_trans[0] for elem in t_diff_trans):
		print("generalized t = " + str(t_diff_trans[0]))
	else:
		t = stats.t.interval(0.95, len(t_diff_trans)-1, loc=np.mean(t_diff_trans), scale=stats.sem(t_diff_trans))
		print("generalized t = " + str(t[0]))
	

	