import os
import sys
import cv2
import shutil
import pathlib2
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import copyfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(filename)s | %(lineno)d | %(message)s',
                              '%m-%d-%Y %H:%M:%S')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

__dir__ = pathlib2.Path(__file__).absolute().parent
data_dir = __dir__.parent / 'data'  # directory containing all datasets and processed datasets
dataset_dir = data_dir / 'CIFAR-10-C'  # directory containing original cifar-10-c npy files
output_dir = data_dir / 'cifar-10-c-images'  # directory containing processed cifar-10-c images


def main(original_dataset: pathlib2.Path, output_path: pathlib2.Path):
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    cifar_10_c_files = os.listdir(original_dataset)
    cifar_10_c_files.remove("labels.npy")
    logger.info("Corruption files:\n" + str(cifar_10_c_files))
    label_path = original_dataset / "labels.npy"
    copyfile(label_path, output_path / "labels.npy")
    labels = np.load(label_path)
    label_df = pd.DataFrame(data={'filename': [f"{label}.png" for label in range(len(labels))], 'label': labels})
    for npy_file in tqdm(cifar_10_c_files):
        corruption_type = npy_file.split('.')[0]
        folder_path = output_path / corruption_type
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)
        data_file_path = original_dataset / npy_file
        data = np.load(data_file_path)
        for i, img in enumerate(data):
            save_path = os.path.join(folder_path, f"{i}.png")
            try:
                cv2.imwrite(save_path, img)
            except Exception as e:
                logger.error(e)
                logger.error(npy_file)
                logger.error(img)
        label_df.to_csv(folder_path / "labels.csv")


if __name__ == '__main__':
    main(dataset_dir, output_dir)
