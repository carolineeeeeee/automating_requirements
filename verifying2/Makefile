download_cifar10:
	if [ ! -e ./data/cifar-10-python.tar.gz -a ! -e ./data/cifar-10-python ]; then wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -O ./data/cifar-10-python.tar.gz; fi
	if [ -e ./data/cifar-10-python.tar.gz -a ! -e ./data/cifar-10-python ]; then tar xvzf ./data/cifar-10-python.tar.gz -C ./data; fi
download_cifar10_c:
	if [ ! -e ./data/CIFAR-10-C.tar -a ! -e ./data/CIFAR-10-C ]; then wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar -O ./data/CIFAR-10-C.tar; fi
	if [ -e ./data/CIFAR-10-C.tar -a ! -e ./data/CIFAR-10-C ]; then tar -xvf ./data/CIFAR-10-C.tar -C ./data; fi

preprocess_cifar10_pytorch:
	python ./prepare/preprocess_cifar10_pytorch.py