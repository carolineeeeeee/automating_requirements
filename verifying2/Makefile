imagenet_data_dir=${PWD}/data/imagenet
cifar10_data_dir=${PWD}/data/cifar10

setup_dir:
	mkdir -p ${imagenet_data_dir}
	# mkdir -p ./data/cifar10

download_cifar10:
	if [ ! -e ./data/cifar-10-python.tar.gz -a ! -e ./data/cifar-10-python ]; then wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O ./data/cifar-10-python.tar.gz; fi
	if [ -e ./data/cifar-10-python.tar.gz -a ! -e ./data/cifar-10-python ]; then tar xvzf ./data/cifar-10-python.tar.gz -C ./data; fi

download_cifar10_c:
	if [ ! -e ./data/CIFAR-10-C.tar -a ! -e ./data/CIFAR-10-C ]; then wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar -O ./data/CIFAR-10-C.tar; fi
	if [ -e ./data/CIFAR-10-C.tar -a ! -e ./data/CIFAR-10-C ]; then tar -xvf ./data/CIFAR-10-C.tar -C ./data; fi

download_imagenet_mapping: setup_dir
	if [ ! -e ${imagenet_data_dir}/MSCOCO_to_ImageNet_category_mapping.txt ]; then wget -O ${imagenet_data_dir}/MSCOCO_to_ImageNet_category_mapping.txt https://raw.githubusercontent.com/rgeirhos/generalisation-humans-DNNs/master/16-class-ImageNet/MSCOCO_to_ImageNet_category_mapping.txt; fi
	if [ ! -e ${imagenet_data_dir}/synset_words.txt ]; then wget -O ${imagenet_data_dir}/synset_words.txt https://raw.githubusercontent.com/HoldenCaulfieldRye/caffe/master/data/ilsvrc12/synset_words.txt; fi

download_imagenet_bbox: setup_dir
	if [ ! -e ${imagenet_data_dir}/ILSVRC2012_bbox_val_v3.tgz -a ! -e ${imagenet_data_dir}/val ]; then wget -O ${imagenet_data_dir}/ILSVRC2012_bbox_val_v3.tgz https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz; fi
	if [ -e ${imagenet_data_dir}/ILSVRC2012_bbox_val_v3.tgz -a ! -e ${imagenet_data_dir}/val ]; then tar -C ${imagenet_data_dir} -xzf ${imagenet_data_dir}/ILSVRC2012_bbox_val_v3.tgz; fi

download_imagenet_val_img: setup_dir
	if [ ! -e ${imagenet_data_dir}/ILSVRC2012_img_val.tar -a ! -e ${imagenet_data_dir}/imgs ]; then wget -O ${imagenet_data_dir}/ILSVRC2012_img_val.tar https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar; fi
	if [ -e ${imagenet_data_dir}/ILSVRC2012_img_val.tar -a ! -e ${imagenet_data_dir}/imgs ]; then mkdir -p ${imagenet_data_dir}/imgs && tar xf ${imagenet_data_dir}/ILSVRC2012_img_val.tar -C ${imagenet_data_dir}/imgs; fi

produce_imagenet_label: setup_dir download_imagenet_mapping download_imagenet_val_img
	if [ ! -e ${imagenet_data_dir}/imagenet_label.sh ]; then wget -O ${imagenet_data_dir}/imagenet_label.sh https://pjreddie.com/media/files/imagenet_label.sh; fi
	if [ ! -e ${imagenet_data_dir}/labelled ]; then cd ${imagenet_data_dir} && bash ${imagenet_data_dir}/imagenet_label.sh; fi
	python ./prepare/prepare_imagenet.py

preprocess_cifar10_pytorch:
	python ./prepare/preprocess_cifar10_pytorch.py

preprocess_cifar10c:
	python ./prepare/preprocess_cifar10_c.py

download: download_cifar10 download_cifar10_c download_imagenet_mapping download_imagenet_bbox download_imagenet_val_img

preprocess: preprocess_cifar10_pytorch preprocess_cifar10c produce_imagenet_label

all: download preprocess

clean:
	bash ./utils/clean.sh