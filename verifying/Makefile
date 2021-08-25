imagenet_data_dir=${PWD}/data/imagenet
cifar10_data_dir=${PWD}/data/cifar10

setup_dir:
	mkdir -p ${imagenet_data_dir}
	# mkdir -p ./data/cifar10
	mkdir ${PWD}/recognition_files
	mkdir ${PWD}/bootstrap_files

# download cifar10 dataset to ./data/cifar-10-python.tar.gz
# uncompress the dataset file to ./data/cifar-10-batches-py in pickle format
download_cifar10:
	if [ ! -e ./data/cifar-10-python.tar.gz -a ! -e ./data/cifar-10-python ]; then wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O ./data/cifar-10-python.tar.gz; fi
	if [ -e ./data/cifar-10-python.tar.gz -a ! -e ./data/cifar-10-python ]; then tar xvzf ./data/cifar-10-python.tar.gz -C ./data; fi

# download cifar10C dataset file to ./data/CIFAR-10-C.tar
# uncompress the dataset file to ./data/CIFAR-10-C in .npy format
download_cifar10_c:
	if [ ! -e ./data/CIFAR-10-C.tar -a ! -e ./data/CIFAR-10-C ]; then wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar -O ./data/CIFAR-10-C.tar; fi
	if [ -e ./data/CIFAR-10-C.tar -a ! -e ./data/CIFAR-10-C ]; then tar -xvf ./data/CIFAR-10-C.tar -C ./data; fi


# save file for imagenet label mapping to ./data/imagenet, the files are MSCOCO_to_ImageNet_category_mapping.txt and synset_words.txt
download_imagenet_mapping: setup_dir
	if [ ! -e ${imagenet_data_dir}/MSCOCO_to_ImageNet_category_mapping.txt ]; then wget -O ${imagenet_data_dir}/MSCOCO_to_ImageNet_category_mapping.txt https://raw.githubusercontent.com/rgeirhos/generalisation-humans-DNNs/master/16-class-ImageNet/MSCOCO_to_ImageNet_category_mapping.txt; fi
	if [ ! -e ${imagenet_data_dir}/synset_words.txt ]; then wget -O ${imagenet_data_dir}/synset_words.txt https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt; fi


# download imagenet bounding box info to ./data/imagenet/ILSVRC2012_bbox_val_v3.tgz
# uncompress ILSVRC2012_bbox_val_v3.tgz to ./data/imagenet/val
download_imagenet_bbox: setup_dir
	if [ ! -e ${imagenet_data_dir}/ILSVRC2012_bbox_val_v3.tgz -a ! -e ${imagenet_data_dir}/val ]; then wget -O ${imagenet_data_dir}/ILSVRC2012_bbox_val_v3.tgz https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz; fi
	if [ -e ${imagenet_data_dir}/ILSVRC2012_bbox_val_v3.tgz -a ! -e ${imagenet_data_dir}/val ]; then tar -C ${imagenet_data_dir} -xzf ${imagenet_data_dir}/ILSVRC2012_bbox_val_v3.tgz; fi

# download imagenet validation dataset to ./data/imagenet/ILSVRC2012_img_val.tar
# uncompress the dataset images into ./data/imagenet/imgs
download_imagenet_val_img: setup_dir
	if [ ! -e ${imagenet_data_dir}/ILSVRC2012_img_val.tar -a ! -e ${imagenet_data_dir}/imgs ]; then wget -O ${imagenet_data_dir}/ILSVRC2012_img_val.tar https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar; fi
	if [ -e ${imagenet_data_dir}/ILSVRC2012_img_val.tar -a ! -e ${imagenet_data_dir}/imgs ]; then mkdir -p ${imagenet_data_dir}/imgs && tar xf ${imagenet_data_dir}/ILSVRC2012_img_val.tar -C ${imagenet_data_dir}/imgs; fi

# download imagenet_label.sh file for labelling images
# run `imagenet_label.sh`, this depends on the files `./data/imagenet/val/*.xml`, so `make download_imagenet_bbox` is a dependency, which generates `val` directory
# Note: if your network isn't good, you may fail to download `imagenet_label.sh`, an empty file could be saved to your file system, no error will be raised but the rest of the code will fail. So if something went wrong here, check if this file is empty.
# run `./prepare/prepare_imagenet.py` to produce all kinds of imagenet label mapping in json and csv format for easier access in the future, these files are saved in `./data/imagenet/info`
produce_imagenet_label: setup_dir download_imagenet_mapping download_imagenet_bbox download_imagenet_val_img
	if [ ! -e ${imagenet_data_dir}/imagenet_label.sh ]; then wget -O ${imagenet_data_dir}/imagenet_label.sh https://pjreddie.com/media/files/imagenet_label.sh; fi
	if [ ! -e ${imagenet_data_dir}/labelled ]; then cd ${imagenet_data_dir} && bash ${imagenet_data_dir}/imagenet_label.sh; fi
	python ./prepare/prepare_imagenet.py

# cifar10 dataset gives us dataset saved in pickle format, it's not easy to read and reshape the arrays and save them as images
# pytorch provides a easier way of loading images and labels
# this script will load images and labels in tensors using a dataloader provided by torchvision and save them as actual images into another directory, ./data/cifar10_pytorch
preprocess_cifar10_pytorch:
	python ./prepare/preprocess_cifar10_pytorch.py

# cifar10c dataset is also saved as npy files, can be downloaded using "make download_cifar10_c"
# to make loading images easier, we run ./prepare/preprocess_cifar10_c.py to extract images from .npy files and save the labels as labels.csv in each corruption dataset
# images saved in ./data/cifar-10-c-images, each corruption has a folder, and a labels.csv in the folder
preprocess_cifar10c:
	python ./prepare/preprocess_cifar10_c.py

# run all downloading scripts
download: download_cifar10 download_cifar10_c download_imagenet_mapping download_imagenet_bbox download_imagenet_val_img

# run all preparing/preprocessing scripts
preprocess: preprocess_cifar10_pytorch preprocess_cifar10c produce_imagenet_label

# run everything
# if something went wrong, you can run the commands above one by one in order, then check in which step the desired files are not generated accoring to the comments
all: setup_dir download preprocess

# clean all of the files generated from the scripts above
clean:
	bash ./utils/clean.sh
