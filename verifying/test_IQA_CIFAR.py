import matlab.engine
import numpy as np
import cv2
import pickle
import time
from skimage.color import rgb2gray
import os
import matplotlib.pyplot as plt

# obtain CIFAR-10-C from https://zenodo.org/record/2535967#.YO4bbxNKglI
# obtain CIFAR-10 from http://www.cs.toronto.edu/~kriz/cifar.html

#LOAD_OR_READ = 'load'
LOAD_OR_READ = 'display'


with open('cifar-10-batches-py/test_batch', 'rb') as fo:
	test_orig_images_dict = pickle.load(fo, encoding='bytes')
	test_orig_data = test_orig_images_dict[b'data']
	test_orig_labels = test_orig_images_dict[b'labels']


label_files = np.load('CIFAR-10-C/labels.npy')
m = len(test_orig_labels)
different = False
for i in range(m):
	if test_orig_labels[i] == label_files[i] \
	and test_orig_labels[i] == label_files[10000 + i] \
	and test_orig_labels[i] == label_files[20000 + i] \
	and test_orig_labels[i] == label_files[30000 + i] \
	and test_orig_labels[i] == label_files[10000 + i]:
		continue
	else:
		print("not the same: " + str(i))
		different = True
if not different:
	print("all good")

#print(label_files[:10000])
#print(label_files[10000:20000])
#print(label_files[20000:30000])
#print(label_files[30000:40000])
#print(label_files[40000:50000])
#exit()

#Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue
def convert_array_to_img(index):
	image_array = test_orig_data[index]
	#print(image_array[:1024].shape)
	#print(image_array[1024:2048].shape)
	#print(image_array[2048:].shape)
	red = np.reshape(image_array[:1024], (32, 32))
	green = np.reshape(image_array[1024:2048], (32, 32))
	blue = np.reshape(image_array[2048:], (32, 32))

	final = np.stack((red, green, blue), axis=-1)
	#print(final.shape)

	return final

if LOAD_OR_READ == 'load':
	#print(test_orig_labels)
	brightness_images = np.load('CIFAR-10-C/brightness.npy')
	contrast_images = np.load('CIFAR-10-C/contrast.npy')
	defocus_blur_images = np.load('CIFAR-10-C/defocus_blur.npy')
	elastic_images = np.load('CIFAR-10-C/elastic_transform.npy')
	fog_images = np.load('CIFAR-10-C/fog.npy')
	frost_images = np.load('CIFAR-10-C/frost.npy')
	gaussian_blur_images = np.load('CIFAR-10-C/gaussian_blur.npy')
	gaussian_noise_images = np.load('CIFAR-10-C/gaussian_noise.npy')
	glass_blur_images = np.load('CIFAR-10-C/glass_blur.npy')
	impulse_noise_images = np.load('CIFAR-10-C/impulse_noise.npy')
	jpeg_compression_images = np.load('CIFAR-10-C/jpeg_compression.npy')
	motion_blur_images = np.load('CIFAR-10-C/motion_blur.npy')
	pixelate_images = np.load('CIFAR-10-C/pixelate.npy')
	saturate_images = np.load('CIFAR-10-C/saturate.npy')
	shot_noise_images = np.load('CIFAR-10-C/shot_noise.npy')
	snow_images = np.load('CIFAR-10-C/snow.npy')
	spatter_images = np.load('CIFAR-10-C/spatter.npy')
	speckle_noise_images = np.load('CIFAR-10-C/speckle_noise.npy')
	zoom_blur_images = np.load('CIFAR-10-C/zoom_blur.npy')

	IQA = 'vif/vifvec_release'
	IQA_PATH = 'image-quality-tools/metrix_mux/metrix/' + IQA + '/'
	matlabPyrToolsPath = "image-quality-tools/metrix_mux/metrix/vif/vifvec_release/matlabPyrTools"

	eng = matlab.engine.start_matlab()
	eng.addpath(IQA_PATH, nargout=0)
	#eng.addpath(SSIM_PATH, nargout=0)
	eng.addpath(matlabPyrToolsPath, nargout=0)
	eng.addpath(matlabPyrToolsPath+ '/MEX', nargout=0)

	brightness_IQA = {}
	contrast_IQA = {}
	defocus_blur_IQA = {}
	elastic_IQA = {}
	fog_IQA = {}
	frost_IQA = {}
	gaussian_blur_IQA = {}
	gaussian_noise_IQA = {}
	glass_blur_IQA = {}
	impulse_noise_IQA = {}
	jpeg_compression_IQA = {}
	motion_blur_IQA = {}
	pixelate_IQA = {}
	saturate_IQA = {}
	shot_noise_IQA = {}
	snow_IQA = {}
	spatter_IQA = {}
	speckle_noise_IQA = {}
	zoom_blur_IQA = {}

	for i in range(m):
		#time_start = time.process_time()
		orig_img = convert_array_to_img(i)
		orig_g = np.float32(rgb2gray(orig_img))
		print(str(i) + '/10000')

		for j in [0, 10000, 20000, 30000, 40000]:
			#brightness
			brightness_g = np.float32(rgb2gray(brightness_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(brightness_g).tolist()))
			#print('brightness: ' + str(IQA_score))
			brightness_IQA[j+i] = IQA_score

			#contrast
			contrast_g = np.float32(rgb2gray(contrast_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(contrast_g).tolist()))
			#print('contrast: ' + str(IQA_score))
			contrast_IQA[j+i] = IQA_score

			#defocus_blur
			defocus_blur_g = np.float32(rgb2gray(defocus_blur_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(defocus_blur_g).tolist()))
			#print('defocus_blur: ' + str(IQA_score))
			defocus_blur_IQA[j+i] = IQA_score

			#elastic
			elastic_g = np.float32(rgb2gray(elastic_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(elastic_g).tolist()))
			#print('elastic: ' + str(IQA_score))
			elastic_IQA[j+i] = IQA_score

			#fog
			fog_g = rgb2gray(fog_images[j+i])
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(fog_g).tolist()))
			#print('fog: ' + str(IQA_score))
			fog_IQA[j+i] = IQA_score

			#frost
			frost_g = np.float32(rgb2gray(frost_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(frost_g).tolist()))
			#print('frost: ' + str(IQA_score))
			frost_IQA[j+i] = IQA_score

			#gaussian_blur
			gaussian_blur_g = np.float32(rgb2gray(gaussian_blur_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(gaussian_blur_g).tolist()))
			#print('gaussian_blur: ' + str(IQA_score))
			gaussian_blur_IQA[j+i] = IQA_score

			#gaussian_noise
			gaussian_noise_g = np.float32(rgb2gray(gaussian_noise_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(gaussian_noise_g).tolist()))
			#print('gaussian_noise: ' + str(IQA_score))
			gaussian_noise_IQA[j+i] = IQA_score

			#glass_blur
			glass_blur_g = np.float32(rgb2gray(glass_blur_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(glass_blur_g).tolist()))
			#print('glass_blur: ' + str(IQA_score))
			glass_blur_IQA[j+i] = IQA_score

			#impulse_noise
			impulse_noise_g = np.float32(rgb2gray(impulse_noise_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(impulse_noise_g).tolist()))
			#print('impulse_noise: ' + str(IQA_score))
			impulse_noise_IQA[j+i] = IQA_score

			#jpeg_compression
			jpeg_compression_g = np.float32(rgb2gray(jpeg_compression_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(jpeg_compression_g).tolist()))
			#print('jpeg_compression: ' + str(IQA_score))
			jpeg_compression_IQA[j+i] = IQA_score

			#motion_blur
			motion_blur_g = np.float32(rgb2gray(motion_blur_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(motion_blur_g).tolist()))
			#print('motion_blur: ' + str(IQA_score))
			motion_blur_IQA[j+i] = IQA_score

			#pixelate
			pixelate_g = np.float32(rgb2gray(pixelate_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(pixelate_g).tolist()))
			#print('pixelate: ' + str(IQA_score))
			pixelate_IQA[j+i] = IQA_score

			#saturate
			saturate_g = np.float32(rgb2gray(saturate_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(saturate_g).tolist()))
			#print('saturate: ' + str(IQA_score))
			saturate_IQA[j+i] = IQA_score

			#shot_noise
			shot_noise_g = np.float32(rgb2gray(shot_noise_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(shot_noise_g).tolist()))
			#print('shot_noise: ' + str(IQA_score))
			shot_noise_IQA[j+i] = IQA_score

			#snow
			snow_g = np.float32(rgb2gray(snow_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(snow_g).tolist()))
			#print('snow: ' + str(IQA_score))
			snow_IQA[j+i] = IQA_score

			#spatter
			spatter_g = np.float32(rgb2gray(spatter_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(spatter_g).tolist()))
			#print('spatter: ' + str(IQA_score))
			spatter_IQA[j+i] = IQA_score

			#speckle
			speckle_noise_g = np.float32(rgb2gray(speckle_noise_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(speckle_noise_g).tolist()))
			#print('speckle: ' + str(IQA_score))
			speckle_noise_IQA[j+i] = IQA_score

			#zoom_blur
			zoom_blur_g = np.float32(rgb2gray(zoom_blur_images[j+i]))
			IQA_score = eng.vifvec2_layers(matlab.double(np.asarray(orig_g).tolist()), matlab.double(np.asarray(zoom_blur_g).tolist()))
			#print('zoom_blur: ' + str(IQA_score))
			zoom_blur_IQA[j+i] = IQA_score

		#time_elapsed = (time.process_time() - time_start)
		#print('time spent: ' + str(time_elapsed))
		#break

	if not os.path.exists('CIFAR_10_C_IQA'):
		os.makedirs('CIFAR_10_C_IQA')

	pickle.dump(brightness_IQA, open('CIFAR_10_C_IQA/brightness_IQA.pkl', 'wb'))
	pickle.dump(contrast_IQA, open('CIFAR_10_C_IQA/contrast_IQA.pkl', 'wb'))
	pickle.dump(defocus_blur_IQA, open('CIFAR_10_C_IQA/defocus_blur_IQA.pkl', 'wb'))
	pickle.dump(elastic_IQA, open('CIFAR_10_C_IQA/elastic_IQA.pkl', 'wb'))
	pickle.dump(fog_IQA, open('CIFAR_10_C_IQA/fog_IQA.pkl', 'wb'))
	pickle.dump(frost_IQA, open('CIFAR_10_C_IQA/frost_IQA.pkl', 'wb'))
	pickle.dump(gaussian_blur_IQA, open('CIFAR_10_C_IQA/gaussian_blur_IQA.pkl', 'wb'))
	pickle.dump(gaussian_noise_IQA, open('CIFAR_10_C_IQA/gaussian_noise_IQA.pkl', 'wb'))
	pickle.dump(glass_blur_IQA, open('CIFAR_10_C_IQA/glass_blur_IQA.pkl', 'wb'))
	pickle.dump(impulse_noise_IQA, open('CIFAR_10_C_IQA/impulse_noise_IQA.pkl', 'wb'))
	pickle.dump(jpeg_compression_IQA, open('CIFAR_10_C_IQA/jpeg_compression_IQA.pkl', 'wb'))
	pickle.dump(motion_blur_IQA, open('CIFAR_10_C_IQA/motion_blur_IQA.pkl', 'wb'))
	pickle.dump(pixelate_IQA, open('CIFAR_10_C_IQA/pixelate_IQA.pkl', 'wb'))
	pickle.dump(saturate_IQA, open('CIFAR_10_C_IQA/saturate_IQA.pkl', 'wb'))
	pickle.dump(shot_noise_IQA, open('CIFAR_10_C_IQA/shot_noise_IQA.pkl', 'wb'))
	pickle.dump(snow_IQA, open('CIFAR_10_C_IQA/snow_IQA.pkl', 'wb'))
	pickle.dump(spatter_IQA, open('CIFAR_10_C_IQA/spatter_IQA.pkl', 'wb'))
	pickle.dump(speckle_noise_IQA, open('CIFAR_10_C_IQA/speckle_noise_IQA.pkl', 'wb'))
	pickle.dump(zoom_blur_IQA, open('CIFAR_10_C_IQA/zoom_blur_IQA.pkl', 'wb'))

	eng.quit()

	'''
	brightness_IQA_l = pickle.load(open('CIFAR_10_C_IQA/brightness_IQA.pkl', 'rb'))
	contrast_IQA_l = pickle.load(open('CIFAR_10_C_IQA/contrast_IQA.pkl', 'rb'))
	defocus_blur_IQA_l = pickle.load(open('CIFAR_10_C_IQA/defocus_blur_IQA.pkl', 'rb'))
	elastic_IQA_l = pickle.load(open('CIFAR_10_C_IQA/elastic_IQA.pkl', 'rb'))
	fog_IQA_l = pickle.load(open('CIFAR_10_C_IQA/fog_IQA.pkl', 'rb'))
	frost_IQA_l = pickle.load(open('CIFAR_10_C_IQA/frost_IQA.pkl', 'rb'))
	gaussian_blur_IQA_l = pickle.load(open('CIFAR_10_C_IQA/gaussian_blur_IQA.pkl', 'rb'))
	gaussian_noise_IQA_l = pickle.load(open('CIFAR_10_C_IQA/gaussian_noise_IQA.pkl', 'rb'))
	glass_blur_IQA_l = pickle.load(open('CIFAR_10_C_IQA/glass_blur_IQA.pkl', 'rb'))
	impulse_noise_IQA_l = pickle.load(open('CIFAR_10_C_IQA/impulse_noise_IQA.pkl', 'rb'))
	jpeg_compression_IQA_l = pickle.load(open('CIFAR_10_C_IQA/jpeg_compression_IQA.pkl', 'rb'))
	motion_blur_IQA_l = pickle.load(open('CIFAR_10_C_IQA/motion_blur_IQA.pkl', 'rb'))
	pixelate_IQA_l = pickle.load(open('CIFAR_10_C_IQA/pixelate_IQA.pkl', 'rb'))
	saturate_IQA_l = pickle.load(open('CIFAR_10_C_IQA/saturate_IQA.pkl', 'rb'))
	shot_noise_IQA_l = pickle.load(open('CIFAR_10_C_IQA/shot_noise_IQA.pkl', 'rb'))
	snow_IQA_l = pickle.load(open('CIFAR_10_C_IQA/snow_IQA.pkl', 'rb'))
	spatter_IQA_l = pickle.load(open('CIFAR_10_C_IQA/spatter_IQA.pkl', 'rb'))
	speckle_noise_IQA_l = pickle.load(open('CIFAR_10_C_IQA/speckle_noise_IQA.pkl', 'rb'))
	zoom_blur_IQA_l = pickle.load(open('CIFAR_10_C_IQA/zoom_blur_IQA.pkl', 'rb'))

	for x in [0, 10000, 20000, 30000, 40000]:
		print(x)
		if brightness_IQA_l[x] != brightness_IQA[x]:
			print("brightness is wrong")
		
		if contrast_IQA_l[x] != contrast_IQA[x]:
			print("contrast is wrong")
		
		if defocus_blur_IQA_l[x] != defocus_blur_IQA[x]:
			print("defocus_blur is wrong")

		if elastic_IQA_l[x] != elastic_IQA[x]:
			print("elastic is wrong")
		
		if fog_IQA_l[x] != fog_IQA[x]:
			print("fog is wrong")
		
		if frost_IQA_l[x] != frost_IQA[x]:
			print("frost is wrong")
		
		if gaussian_blur_IQA_l[x] != gaussian_blur_IQA[x]:
			print("gaussian_blur is wrong")
		
		if gaussian_noise_IQA_l[x] != gaussian_noise_IQA[x]:
			print("gaussian_noise is wrong")

		if glass_blur_IQA_l[x] != glass_blur_IQA[x]:
			print("glass_blur is wrong")
		
		if impulse_noise_IQA_l[x] != impulse_noise_IQA[x]:
			print("impulse_noise is wrong")

		if jpeg_compression_IQA_l[x] != jpeg_compression_IQA[x]:
			print("jpeg_compression is wrong")
		
		if motion_blur_IQA_l[x] != motion_blur_IQA[x]:
			print("motion_blur is wrong")

		if pixelate_IQA_l[x] != pixelate_IQA[x]:
			print("pixelate is wrong")

		if saturate_IQA_l[x] != saturate_IQA[x]:
			print("saturate is wrong")

		if shot_noise_IQA_l[x] != shot_noise_IQA[x]:
			print("shot_noise is wrong")

		if snow_IQA_l[x] != snow_IQA[x]:
			print("snow is wrong")

		if spatter_IQA_l[x] != spatter_IQA[x]:
			print("spatter is wrong")
		
		if speckle_noise_IQA_l[x] != speckle_noise_IQA[x]:
			print("speckle_noise is wrong")

		if zoom_blur_IQA_l[x] != zoom_blur_IQA[x]:
			print("zoom_blur is wrong")
	'''
else:
	transf = 'frost'
	transformation_t = 0.89

	transformation_IQA = pickle.load(open('CIFAR_10_C_IQA/'+transf+'_IQA.pkl', 'rb'))
	if transf == 'elastic':
		transformation_images = np.load('CIFAR-10-C/'+transf+'transform.npy')
	else:
		transformation_images = np.load('CIFAR-10-C/'+transf+'.npy')
	max_iqa = 0
	max_img_id = 0
	min_iqa = 1
	min_img_id = 0
	closest = 1
	closest_id = 0
	
	for img in transformation_IQA.keys():

		if transformation_IQA[img] > max_iqa:
			max_iqa = transformation_IQA[img] 
			max_img_id = img

		if transformation_IQA[img] < min_iqa:
			min_iqa = transformation_IQA[img]
			min_img_id = img

		if abs((1-transformation_IQA[img])-transformation_t) < closest:
			closest = abs((1-transformation_IQA[img])-transformation_t)
			closest_id = img

	orig_max = convert_array_to_img(max_img_id % 10000)
	orig_min = convert_array_to_img(min_img_id % 10000)
	orig_closest = convert_array_to_img(closest_id % 10000)
	trans_max = transformation_images[max_img_id]
	trans_min = transformation_images[min_img_id]
	trans_closest = transformation_images[closest_id]

	fig = plt.figure(figsize=(10, 7))
	rows = 3
	columns = 2
	  
	# Adds a subplot at the 1st position
	fig.add_subplot(rows, columns, 1)
	plt.imshow(orig_max)
	plt.axis('off')
	plt.title("original image " + str(max_img_id % 10000) )
	  
	# Adds a subplot at the 2nd position
	fig.add_subplot(rows, columns, 2)
	plt.imshow(trans_max)
	plt.axis('off')
	plt.title(transf + " max IQA " + str(max_img_id)+ ', ' + str(transformation_IQA[max_img_id]))
	  
	# Adds a subplot at the 3rd position
	fig.add_subplot(rows, columns, 3)
	plt.imshow(orig_min)
	plt.axis('off')
	plt.title("orignal image "+ str(min_img_id % 10000))
	  
	# Adds a subplot at the 4th position
	fig.add_subplot(rows, columns, 4)
	plt.imshow(trans_min)
	plt.axis('off')
	plt.title(transf + " min IQA " + str(min_img_id)+ ', ' + str(transformation_IQA[min_img_id]))

	# Adds a subplot at the 3rd position
	fig.add_subplot(rows, columns, 5)
	plt.imshow(orig_closest)
	plt.axis('off')
	plt.title("orignal image " + str(closest_id % 10000))
	  
	# Adds a subplot at the 4th position
	fig.add_subplot(rows, columns, 6)
	plt.imshow(trans_closest)
	plt.axis('off')
	plt.title(transf + " closest IQA to t " + str(closest_id)+ ', ' + str(transformation_IQA[closest_id]))

	plt.show()
	exit()

	

	brightness_IQA = pickle.load(open('CIFAR_10_C_IQA/brightness_IQA.pkl', 'rb'))
	plt.title('IQA distribution for brightness_IQA' + ', range: (' + str(round(max(brightness_IQA.values()), 2)) + ',' + str(round(min(brightness_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in brightness_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	contrast_IQA = pickle.load(open('CIFAR_10_C_IQA/contrast_IQA.pkl', 'rb'))
	plt.title('IQA distribution for contrast_IQA' + ', range: (' + str(round(max(contrast_IQA.values()), 2)) + ',' + str(round(min(contrast_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in contrast_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	defocus_blur_IQA = pickle.load(open('CIFAR_10_C_IQA/defocus_blur_IQA.pkl', 'rb'))
	plt.title('IQA distribution for defocus_blur_IQA' + ', range: (' + str(round(max(defocus_blur_IQA.values()), 2)) + ',' + str(round(min(defocus_blur_IQA.values()),2)) + ')')
	plt.hist(list([i for i in defocus_blur_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	elastic_IQA = pickle.load(open('CIFAR_10_C_IQA/elastic_IQA.pkl', 'rb'))
	plt.title('IQA distribution for elastic_IQA' + ', range: (' + str(round(max(elastic_IQA.values()),2)) + ',' + str(round(min(elastic_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in elastic_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	fog_IQA = pickle.load(open('CIFAR_10_C_IQA/fog_IQA.pkl', 'rb'))
	plt.title('IQA distribution for fog_IQA' + ', range: (' + str(round(max(fog_IQA.values()), 2)) + ',' + str(round(min(fog_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in fog_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	frost_IQA = pickle.load(open('CIFAR_10_C_IQA/frost_IQA.pkl', 'rb'))
	plt.title('IQA distribution for frost_IQA' + ', range: (' + str(round(max(frost_IQA.values()), 2)) + ',' + str(round(min(frost_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in frost_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	gaussian_blur_IQA = pickle.load(open('CIFAR_10_C_IQA/gaussian_blur_IQA.pkl', 'rb'))
	plt.title('IQA distribution for gaussian_blur_IQA' + ', range: (' + str(round(max(gaussian_blur_IQA.values()), 2)) + ',' + str(round(min(gaussian_blur_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in gaussian_blur_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	gaussian_noise_IQA = pickle.load(open('CIFAR_10_C_IQA/gaussian_noise_IQA.pkl', 'rb'))
	plt.title('IQA distribution for gaussian_noise_IQA' + ', range: (' + str(round(max(gaussian_noise_IQA.values()), 2)) + ',' + str(round(min(gaussian_noise_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in gaussian_noise_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	glass_blur_IQA = pickle.load(open('CIFAR_10_C_IQA/glass_blur_IQA.pkl', 'rb'))
	plt.title('IQA distribution for glass_blur_IQA' + ', range: (' + str(round(max(glass_blur_IQA.values()), 2)) + ',' + str(round(min(glass_blur_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in glass_blur_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	impulse_noise_IQA = pickle.load(open('CIFAR_10_C_IQA/impulse_noise_IQA.pkl', 'rb'))
	plt.title('IQA distribution for impulse_noise_IQA' + ', range: (' + str(round(max(impulse_noise_IQA.values()), 2)) + ',' + str(round(min(impulse_noise_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in impulse_noise_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	jpeg_compression_IQA = pickle.load(open('CIFAR_10_C_IQA/jpeg_compression_IQA.pkl', 'rb'))
	plt.title('IQA distribution for jpeg_compression_IQA' + ', range: (' + str(round(max(jpeg_compression_IQA.values()), 2)) + ',' + str(round(min(jpeg_compression_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in jpeg_compression_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	motion_blur_IQA = pickle.load(open('CIFAR_10_C_IQA/motion_blur_IQA.pkl', 'rb'))
	plt.title('IQA distribution for motion_blur_IQA' + ', range: (' + str(round(max(motion_blur_IQA.values()), 2)) + ',' + str(round(min(motion_blur_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in motion_blur_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	pixelate_IQA = pickle.load(open('CIFAR_10_C_IQA/pixelate_IQA.pkl', 'rb'))
	plt.title('IQA distribution for pixelate_IQA' + ', range: (' + str(round(max(pixelate_IQA.values()), 2)) + ',' + str(round(min(pixelate_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in pixelate_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	saturate_IQA = pickle.load(open('CIFAR_10_C_IQA/saturate_IQA.pkl', 'rb'))
	plt.title('IQA distribution for saturate_IQA' + ', range: (' + str(round(max(saturate_IQA.values()), 2)) + ',' + str(round(min(saturate_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in saturate_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	shot_noise_IQA = pickle.load(open('CIFAR_10_C_IQA/shot_noise_IQA.pkl', 'rb'))
	plt.title('IQA distribution for shot_noise_IQA' + ', range: (' + str(round(max(shot_noise_IQA.values()), 2)) + ',' + str(round(min(shot_noise_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in shot_noise_IQA.values() if i <= 1]), bins = 50)
	plt.show()
	
	snow_IQA = pickle.load(open('CIFAR_10_C_IQA/snow_IQA.pkl', 'rb'))
	plt.title('IQA distribution for snow_IQA' + ', range: (' + str(round(max(snow_IQA.values()), 2)) + ',' + str(round(min(snow_IQA.values()),2)) + ')')
	plt.hist(list([i for i in snow_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	spatter_IQA = pickle.load(open('CIFAR_10_C_IQA/spatter_IQA.pkl', 'rb'))
	plt.title('IQA distribution for spatter_IQA' + ', range: (' + str(round(max(spatter_IQA.values()), 2)) + ',' + str(round(min(spatter_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in spatter_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	speckle_noise_IQA = pickle.load(open('CIFAR_10_C_IQA/speckle_noise_IQA.pkl', 'rb'))
	plt.title('IQA distribution for speckle_noise_IQA' + ', range: (' + str(round(max(speckle_noise_IQA.values()), 2)) + ',' + str(round(min(speckle_noise_IQA.values()), 2)) + ')')
	plt.hist(list([i for i in speckle_noise_IQA.values() if i <= 1]), bins = 50)
	plt.show()

	zoom_blur_IQA = pickle.load(open('CIFAR_10_C_IQA/zoom_blur_IQA.pkl', 'rb'))
	plt.title('IQA distribution for zoom_blur_IQA' + ', range: (' + str(round(max(zoom_blur_IQA.values()), 2)) + ',' + str(round(min(zoom_blur_IQA.values()), 2)) + ')' )
	plt.hist(list([i for i in zoom_blur_IQA.values() if i <= 1]), bins = 50)
	plt.show()




