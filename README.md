# Requirements
This is for our submission "Tackling the Unspecifiable: Reliability Requirements forMachine-learned Perception based on Human Performance" and contains three parts. First the two parts include our implementation of the proposed automated method and instructions on how to run them. The third part is the supplementary material of the complete list of safety-related CV-HAZOP entries and the list of the ones applicable to our approach.

## 1. Estimating parameters
Our estimation scripts are in estimating/

**Note:** estimating/image_maniputation.py is originally from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/code/image_manipulation.py. We made minor changes to adapt it to our scripts. 

To run parameter estimation in our paper, follow these steps to set up:
1. Download ILSVRC2012 training set from http://www.image-net.org/challenges/LSVRC/2012/
2. Download experiment results for the five transformations used in the paper (contrast, noise, highpass, lowpass, phase noise) from https://github.com/rgeirhos/generalisation-humans-DNNs/tree/master/raw-data/humans. Create a directory in estimating/ and name it csv_files/, then put the files in estimating/csv_files/.
3. Download VIF and CW-SSIM implementation from 
4. Run get_IQA.py with `python3 get_IQA.py --VIF_PATH VIF_PATH --CW_SSIM_PATH CW_SSIM_PATH --matlabPyrTools_PATH matlabPyrTools_PATH` to obtain the image pair to visual_change score mapping 
(**Note:** this step can be skipped since we also provided our pre-compiled file estimating/all_filename_to_IQA.pkl)
5. Run `python3 collect_results -c CLASS` to obtain the parameters for one specific class.

## 2. Verifying ML models with our reliability requirements
Our testing script is verifying/automated_bootstrap.py
Run `python3 automated_bootstrap.py -h` to see instructions for input arguments.

To run the evaluation in the paper, follow these steps to set up:
1. Clone this repo
2. Download MSCOCO_to_ImageNet_category_mapping.txt from https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/16-class-ImageNet/MSCOCO_to_ImageNet_category_mapping.txt, put this file in verifying/ of your local copy of this repository.
3. Download synset_words.txt from https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt, put this file in verifying/ of your local copy of this repository.
4. Install Darknet following instructions here: https://pjreddie.com/darknet/install/
5. Obtain ILSVRC2012 validation images following the instructions in the "Validating On ImageNet" section here: https://pjreddie.com/darknet/imagenet/. Make sure that this file inet.val.list is generated and copy it to verifying/. 
6. From the same page as 2, download the pretrained weights for AlexNet, VGG-16, Darknet19, Darknet53 448x448, Resnet 50 and ResNeXt 50. Put all the weights in the darknet directory.

In the directory verifying/ we also provide bootstrapping and detection result files from the experiments conducted for the evaluation section in the paper. Simply complete steps 1-3 from above and then run `python3 automated_bootstrap.txt -load_existing` to reproduce the results of our evaluation experiments. 

To re-run bootstrapping, complete all the steps then run `python3 automated_bootstrap.txt -d path_to_darknet -i path_to_inet.val.list -n number_of_batches -s number_of_images_per_batch`. 

## 3. Supplementary materials
### 3.1 CV-HAZOP entries 
### 3.1 Extended experiment results