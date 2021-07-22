import os
import cv2
import numpy as np
from tqdm import tqdm
from shutil import copyfile


if __name__ == '__main__':
    original_data_path = "/home/huakun/Downloads/CIFAR-10-C"
    save_dir = "/home/huakun/Documents/summer-research/automating_requirements/verifying/cifar10_c_data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    original_files = os.listdir(original_data_path)
    original_files.remove("labels.npy")
    print(original_files)
    copyfile(os.path.join(original_data_path, "labels.npy"), os.path.join(save_dir, "labels.npy"))
    for npy_file in tqdm(original_files):
        folder_name = npy_file.split('.')[0]
        folder_path = os.path.join(save_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        data_file_path = os.path.join(original_data_path, npy_file)
        data = np.load(data_file_path)
        for i, img in enumerate(data):
            save_path = os.path.join(folder_path, f"{i}.png")
            try:
                cv2.imwrite(save_path, img)
            except Exception as e:
                print(e)
                print(npy_file)
                print(img)
