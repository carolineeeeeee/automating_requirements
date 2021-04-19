import tarfile
import os
import pickle
from os import listdir
from os.path import isfile, join
import csv
import re
import matlab.engine
from image_manipulation import *
import getopt
import argparse

csv_files_dir = './csv_files/'
VIF_PATH = ''
SSIM_PATH = ''
matlabPyrToolsPath = ''

def parse_csv():
	csv_files = [f for f in listdir(csv_files_dir) if isfile(join(csv_files_dir, f)) and 'csv' in f]

	ds_name_to_imgs = {}

	for csv_name in csv_files:
		with open(csv_files_dir + csv_name, newline='') as csv_file:
			raw_data = csv.reader(csv_file)
			i = 0
			for row in raw_data:
				filename = row[-1]
				filename_match = re.findall('((n\d+)_\d+)\..+', filename)
				if filename_match != []:
					if filename_match[0][1] not in ds_name_to_imgs:
						ds_name_to_imgs[filename_match[0][1]] = {}
					if filename_match[0][0] not in ds_name_to_imgs[filename_match[0][1]].keys():
						ds_name_to_imgs[filename_match[0][1]][filename_match[0][0]] = []
					ds_name_to_imgs[filename_match[0][1]][filename_match[0][0]].append(csv_name)
	return ds_name_to_imgs

def collect_img_names(orig_names, csv_files):
	result = []
	for csv_name in csv_files:
		with open(csv_files_dir + csv_name, newline='') as csv_file:
			raw_data = [row[-1] for row in csv.reader(csv_file)]
			result += [i for e in orig_names for i in raw_data if e[:-4] in i]
	return result

def handle_transformation_and_IQA(img_names, tar_name):
	img_name_to_IQA = {}
	i = 0
	filename_to_IQA =pickle.load(open('all_filename_to_IQA.pkl', 'rb'))
	for img_name in img_names:
		i += 1
		print(str(i) + ': ' + img_name)
		filename_match = re.findall('.*_(.*)_.*_(.*)_.*_.*_(.*_.*)\.', img_name)
		transformation = filename_match[0][0]
		parameter = filename_match[0][1]
		orig_img = filename_match[0][2]
		img = imload_rgb('./' + tar_name[:-4] + '/' + orig_img + '.JPEG')
		if transformation == 'rot':
			# read the dict and check if name exists
			if img_name in filename_to_IQA.keys():
				continue
			#TODO: bypassing rotations now
			if parameter == '0':
				print("1")
				img_name_to_IQA[img_name] = 1
			if parameter == '90':
				img2 = rotate90(img)
				SSIM_index = eng.cwssim_index(matlab.double(np.asarray(img*255).tolist()), matlab.double(np.asarray(img2*255).tolist()), float(6), float(16), float(0), float(0))
				print(SSIM_index)
				img_name_to_IQA[img_name] = SSIM_index
				#save_img(img2* 255, "test_image_rotation90", True)
			if parameter == '180':
				img2 = rotate180(img)
				SSIM_index = eng.cwssim_index(matlab.double(np.asarray(img*255).tolist()), matlab.double(np.asarray(img2*255).tolist()), float(6), float(16), float(0), float(0))
				print(SSIM_index)
				img_name_to_IQA[img_name] = SSIM_index
				#save_img(img2* 255, "test_image_rotation180", True)
			if parameter == '270':
				img2 = rotate270(img) 
				SSIM_index = eng.cwssim_index(matlab.double(np.asarray(img*255).tolist()), matlab.double(np.asarray(img2*255).tolist()), float(6), float(16), float(0), float(0))
				print(SSIM_index)
				img_name_to_IQA[img_name] = SSIM_index
				#save_img(img2* 255, "test_image_rotation270", True)
		if transformation == 'con':
			parameter = int(parameter[1:])/100
			img_low_contrast = grayscale_contrast(image=img, contrast_level=parameter)
			#save_img(img_low_contrast * 255, "test_image_low_contrast", True)

			VIF_value = eng.vifvec(matlab.double(np.asarray(img*255).tolist()), matlab.double(np.asarray(img_low_contrast*255).tolist()))
			img_name_to_IQA[img_name] = VIF_value
			#SSIM_value = eng.ssim_index(matlab.double(np.asarray(img*255).tolist()), matlab.double(np.asarray(img_low_contrast*255).tolist()))
			#img_name_to_IQA[img_name] = SSIM_value

		if transformation == 'nse':
			noise_width = float(parameter)
			rng = np.random.RandomState(seed=42)
			img_noisy = uniform_noise(image=img, width=noise_width,contrast_level=1,rng=rng)
			#save_img(img_noisy* 255, "test_image_noisy", True)

			VIF_value = eng.vifvec(matlab.double(np.asarray(img*255).tolist()), matlab.double(np.asarray(img_noisy*255).tolist()))
			img_name_to_IQA[img_name] = VIF_value
		
		if transformation == 'ps':
			width = int(parameter)
			img_phase_scrambling = cv2.cvtColor(phase_scrambling(img, width).astype('float32'), cv2.COLOR_BGR2GRAY)
			VIF_value = eng.vifvec(matlab.double(np.asarray(img*255).tolist()), matlab.double(np.asarray(img_phase_scrambling*255).tolist()))
			img_name_to_IQA[img_name] = VIF_value
		if transformation == 'lp':
			if parameter == 'inf':
				img_name_to_IQA[img_name] = 1
			else:
				std =float(parameter)
				img_lowpass = low_pass_filter(img, std) 
				VIF_value = eng.vifvec(matlab.double(np.asarray(img*255).tolist()), matlab.double(np.asarray(img_lowpass*255).tolist()))
				img_name_to_IQA[img_name] = VIF_value

		if transformation == 'hp':
			if parameter == 'inf':
				img_name_to_IQA[img_name] = 1
			else:
				std =float(parameter)
				img_highpass = high_pass_filter(img, std) 
				VIF_value = eng.vifvec(matlab.double(np.asarray(img*255).tolist()), matlab.double(np.asarray(img_highpass*255).tolist()))
				img_name_to_IQA[img_name] = VIF_value

	return img_name_to_IQA

def usage():
	parser=argparse.ArgumentParser(
	description='''This is the script for mapping visual_change scores to images in the experiment. All three paths are required. ''')
	parser.add_argument('--VIF_PATH', type=str, help='path to VIF matlab implementation')
	parser.add_argument('--CW_SSIM_PATH', type=str, help='path to CW_SSIM matlab implementation')
	parser.add_argument('--matlabPyrTools_PATH', type=str, help='path to matlab library matlabPyrTools required by VIF')
	
	args=parser.parse_args()

if __name__ == '__main__':
	try:
		opts, args = getopt.getopt(sys.argv[1:], "h", ["help", "VIF_PATH=", "CW_SSIM_PATH=", "matlabPyrTools_PATH="])

	except getopt.GetoptError as err:
		print(err)  
		usage()
		sys.exit(2)

	for o,a in opts:
		if o in ("-h", "--help"):
			usage()
			print("help")
			sys.exit()
		elif o in ("--VIF_PATH"):
			VIF_PATH = a
		elif o in ("--CW_SSIM_PATH"):
			SSIM_PATH = a
		elif o in ("--matlabPyrTools_PATH"):
			matlabPyrToolsPath = a
			
		else:
			assert False, "unhandled option"
			print("invalid argument\n")
			usage()
			sys.exit()
	if VIF_PATH == '':
		print("Missing path to VIF")
		usage()
		sys.exit()
	if SSIM_PATH == '':
		print("Missing path to CW_SSIM")
		usage()
		sys.exit()
	if matlabPyrToolsPath == '':
		print("Missing path to matlabPyrTools")
		usage()
		sys.exit()


	# images included in the experiments
	name_list = ['n02027492', 'n04560804', 'n02091635', 'n02107574', 'n02100735', 'n03100240', 'n03796401', 'n02056570', 'n02102318', 'n01560419', 'n02095570', 'n02088238', 'n02025239', 'n02102480', 'n01616318', 'n01622779', 'n02018207', 'n02096294', 'n03041632', 'n02097047', 'n02093428', 'n03937543', 'n03344393', 'n02097209', 'n02814533', 'n02106662', 'n02088094', 'n01558993', 'n02123159', 'n01582220', 'n02106550', 'n01843065', 'n02110185', 'n02100583', 'n02097658', 'n02113978', 'n01817953', 'n03376595', 'n01795545', 'n03792782', 'n01601694', 'n02107683', 'n02087046', 'n04429376', 'n02037110', 'n02105251', 'n02090721', 'n02090622', 'n02823428', 'n02094258', 'n02085782', 'n02093859', 'n02112706', 'n01855032', 'n02097298', 'n02097130', 'n02110627', 'n04285008', 'n04111531', 'n02105855', 'n02100877', 'n01796340', 'n02113799', 'n04548280', 'n02013706', 'n02104365', 'n02018795', 'n02028035', 'n01614925', 'n02104029', 'n02095314', 'n02091134', 'n02835271', 'n02011460', 'n02090379', 'n02002556', 'n02108422', 'n02099601', 'n02088466', 'n02094114', 'n04557648', 'n02111500', 'n02109525', 'n02951358', 'n01829413', 'n01855672', 'n02099267', 'n03085013', 'n02086646', 'n02095889', 'n02123394', 'n02504458', 'n02107312', 'n02134084', 'n02123045', 'n04467665', 'n02009912', 'n02099712', 'n02089973', 'n02092339', 'n02105056', 'n02106166', 'n02108551', 'n02109961', 'n02093991', 'n02091032', 'n02086910', 'n02088364', 'n02111277', 'n03662601', 'n02091831', 'n02125311', 'n02101556', 'n01824575', 'n02098286', 'n04099969', 'n02086079', 'n02102177', 'n02791124', 'n02105162', 'n02091244', 'n02108915', 'n02099849', 'n02113023', 'n04273569', 'n02093647', 'n02504013', 'n02087394', 'n02098105', 'n02033041', 'n02106030', 'n03983396', 'n01530575', 'n01532829', 'n02089867', 'n01843383', 'n02110063', 'n02100236', 'n02091467', 'n02096051', 'n02112350', 'n02124075', 'n01797886', 'n01828970', 'n01833805', 'n02112018', 'n02006656', 'n02111129', 'n02109047', 'n02009229', 'n02105505', 'n01820546', 'n02002724', 'n02089078', 'n01860187', 'n02102040', 'n02102973', 'n02051845', 'n02110806', 'n01819313', 'n01534433', 'n02086240', 'n02134418', 'n02105641', 'n02107142', 'n02123597', 'n04505470', 'n02093256', 'n04579145', 'n02099429', 'n02690373', 'n02107908', 'n02096437', 'n02708093', 'n02017213', 'n02093754', 'n03417042', 'n02106382', 'n02101006', 'n01592084', 'n02132136', 'n01531178', 'n01514859', 'n02092002', 'n02097474', 'n04591713', 'n02110958', 'n02094433', 'n02085936', 'n02007558', 'n02113712', 'n02096585', 'n02133161', 'n01537544', 'n02088632', 'n03196217', 'n02108000', 'n02101388', 'n01798484', 'n04612504', 'n02113624', 'n01818515']

	IQA_files = [f for f in listdir('.') if isfile(f) and 'pkl' in f]
	ds_name_to_imgs = parse_csv()
	eng = matlab.engine.start_matlab()
	eng.addpath(VIF_PATH, nargout=0)
	eng.addpath(SSIM_PATH, nargout=0)
	eng.addpath(matlabPyrToolsPath, nargout=0)
	eng.addpath(matlabPyrToolsPath+ '/MEX', nargout=0)
	
	i = 0
	for name in name_list:
		i += 1
		tar_name = name+'.tar'
		print(tar_name[:-4])
		print("progress: " + str(i) + '/' + str(len(name_list)))
		this_IQA_file = list([f for f in IQA_files if tar_name[:-4] in f])
		
		if tar_name[:-4] in ds_name_to_imgs.keys() and this_IQA_file == []:
			tf = tarfile.open(tar_name)
			if not os.path.exists(tar_name[:-4]):
				os.mkdir(tar_name[:-4])
				tf.extractall(path=tar_name[:-4])
			
			img_files = [f for f in listdir(tar_name[:-4]) if isfile(join(tar_name[:-4], f)) and 'JPEG' in f]
			
			for img in img_files:
				if img[:-5] not in ds_name_to_imgs[tar_name[:-4]].keys():
					os.remove('./' + tar_name[:-4] + '/' + img)
			img_files = [f for f in listdir(tar_name[:-4]) if isfile(join(tar_name[:-4], f)) and 'JPEG' in f]
			csv_files = [f for f in listdir(csv_files_dir) if isfile(join(csv_files_dir, f)) and 'csv' in f]
			exp_img_file_names = collect_img_names(img_files, csv_files)		
			
			IQA_results = handle_transformation_and_IQA(exp_img_file_names, tar_name)
			pickle.dump(IQA_results, open( tar_name[:-4] + "_img_name_to_IQA.pkl", "wb" ) )
			for f in img_files:
				os.remove('./' + tar_name[:-4]+ '/' + f)
			os.rmdir(tar_name[:-4])	
		
	eng.quit()
