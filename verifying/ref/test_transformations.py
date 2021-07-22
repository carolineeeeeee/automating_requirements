from Imagenet_c_transformations import *
import pickle
import matplotlib.pyplot as plt


with open('../cifar-10-batches-py/test_batch', 'rb') as fo:
	test_orig_images_dict = pickle.load(fo, encoding='bytes')
	test_orig_data = test_orig_images_dict[b'data']
	test_orig_labels = test_orig_images_dict[b'labels']

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

one_image = convert_array_to_img(0)

defocus_blured, _ = defocus_blur(one_image, random.choice(range(TRANSFORMATION_LEVEL)))
plt.imshow(defocus_blured / 255) # just for displaying here, converting to [0,1] range
plt.show()

motion_blured, _  = motion_blur(Image.fromarray(one_image), random.choice(range(TRANSFORMATION_LEVEL)))
print(motion_blured)
plt.imshow(motion_blured / 255)# just for displaying here, converting to [0,1] range
plt.show()

jpeg_compressed, _ = jpeg_compression(Image.fromarray(one_image), random.choice(range(TRANSFORMATION_LEVEL)))
print(jpeg_compressed)
plt.imshow(jpeg_compressed)
plt.show()