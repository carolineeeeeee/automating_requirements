import os
import cv2
import argparse
import pathlib2
import numpy as np
from tqdm import tqdm
from shutil import copyfile

__root__ = pathlib2.Path(__file__).absolute().parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Save Cifar10-C")
    parser.add_argument("--original_dataset_path", help="path to origianl cifar-10 dataset")
    parser.add_argument("--cifar_10_c_dataset_path", help="path to Cifar10-C dataset, in .npy format")
    parser.add_argument("--save_dir", default=os.path.join(__root__, "cifar10_c_data"), help="path to save images")
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    cifar_10_c_files = os.listdir(args.cifar_10_c_dataset_path)
    cifar_10_c_files.remove("labels.npy")
    print(cifar_10_c_files)
    label_path = os.path.join(args.cifar_10_c_dataset_path, "labels.npy")
    copyfile(label_path, os.path.join(args.save_dir, "labels.npy"))
    labels = np.load(label_path)
    for npy_file in tqdm(cifar_10_c_files):
        folder_name = npy_file.split('.')[0]
        folder_path = os.path.join(args.save_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        data_file_path = os.path.join(args.cifar_10_c_dataset_path, npy_file)
        data = np.load(data_file_path)
        for i, img in enumerate(data):
            save_path = os.path.join(folder_path, f"{i}.png")
            try:
                cv2.imwrite(save_path, img)
            except Exception as e:
                print(e)
                print(npy_file)
                print(img)
        np.savetxt(os.path.join(folder_path, "labels.txt"), labels)
