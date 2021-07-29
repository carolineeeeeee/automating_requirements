import os
import sys
import shutil
import logging
import pathlib2
import pandas as pd
from tqdm import tqdm
from typing import Union
from src.bootstrap import Bootstrapper
from src.utils import clean_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%m-%d-%Y %H:%M:%S')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.WARNING)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


class Cifar10CBootstrapper(Bootstrapper):
    def __init__(self, num_sample_iter: int, sample_size: int, source: Union[str, pathlib2.Path],
                 destination: Union[str, pathlib2.Path],
                 dataset_info_df: pd.DataFrame, corruption: str):
        super(Cifar10CBootstrapper, self).__init__(num_sample_iter, sample_size, source, destination)
        self.dataset_info_df = dataset_info_df
        self.corruption = corruption

    def _prepare(self):
        if not self.source.exists():
            raise ValueError(f"Source data {self.source} doesn't exist")
        if self.destination.exists():
            shutil.rmtree(self.destination)
        self.destination.mkdir(parents=True, exist_ok=True)

    def run(self):
        self._prepare()
        logger.info("bootstrapping")
        self.data = []
        for i in tqdm(range(self.num_sample_iter)):
            sample_images = self.dataset_info_df.sample(n=self.sample_size, replace=False)
            iteration_path = self.destination / f'batch_{i}'  # output path for current bootstrap iteration
            clean_dir(iteration_path)
            k = 0
            for j, row in sample_images.iterrows():
                cur_row = row
                image_name = cur_row['original_filename']
                image_path = cur_row['original_path']
                output_path = str(iteration_path / image_name)
                os.symlink(image_path, output_path)
                self.data.append({
                    "bootstrap_iter_id": i,
                    "within_iter_id": k,
                    "original_filename": cur_row['original_filename'],
                    "new_filename": image_name,
                    "original_path": cur_row['original_path'],
                    "transformation": self.corruption,
                    "new_path": output_path,
                    "label": cur_row['label']
                })
                k += 1

        self.df = pd.DataFrame(data=self.data)
        return self.df
