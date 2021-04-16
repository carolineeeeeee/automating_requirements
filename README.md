# Requirements
This is for submission blah and contains three parts
## 1. Estimating parameters

## 2. Verifying ML models with our reliability requirements
Our testing script is verifying/automated_bootstrap.py
Run `python3 automated_bootstrap.py -h` to see instructions for input arguments.

To run the evaluation in the paper, follow these steps to set up:
1. Clone this repo
2. Download MSCOCO_to_ImageNet_category_mapping.txt from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/16-class-ImageNet/MSCOCO_to_ImageNet_category_mapping.txt, put this file in verifying/
3. Download synset_words.txt from https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt, put this file in verifying/
4. Install Darknet following instructions here: https://pjreddie.com/darknet/install/
5. Obtain ILSVRC2012 validation images following the instructions in the "Validating On ImageNet" section here: https://pjreddie.com/darknet/imagenet/. Make sure that this file inet.val.list is generated and copy it to verifying/. 
6. From the same page as 2, download the pretrained weights for AlexNet, VGG-16, Darknet19, Darknet53 448x448, Resnet 50 and ResNeXt 50. Put all the weights in the darknet directory.

In the directory verifying/ we also provide bootstrapping and detection result files from the experiments conducted for the evaluation section in the paper. Simply run `python3 automated_bootstrap.txt -r` to reproduce the results of our evaluation experiments. 

To re-run bootstrapping, run `python3 automated_bootstrap.txt -d path_to_darknet -i path_to_ILSVRC2012_validation_images -n number_of_batches -s number_of_images_per_batch`. 

## 3. Supplementary matrials