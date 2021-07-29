import os
import pickle
import pathlib2
from abc import ABC, abstractmethod
from src.bootstrap import Cifar10Bootstrapper
from src.constant import ROOT, GAUSSIAN_NOISE, IQA, IQA_PATH, matlabPyrToolsPath
from src.utils import start_matlab, load_cifar10_data, read_cifar10_ground_truth
from src.evaluate import run_model, estimate_conf_int
from .bootstrap import Cifar10CBootstrapper
from src.job import Job


class Cifar10CJob(Job):
    def __init__(self, source: str, destination: str, num_sample_iter: int, sample_size: int, corruption: str,
                model_name: str, rq_type: str):
        super(Cifar10CJob, self).__init__(source, destination, num_sample_iter, sample_size)
        self.rq_type = rq_type
        dataset_info_df = load_cifar10_data(pathlib2.Path(self.source))
        self.bootstrapper = Cifar10CBootstrapper(num_sample_iter=self.num_sample_iter, sample_size=self.sample_size,
                                                 source=source, destination=destination,
                                                 dataset_info_df=dataset_info_df, corruption=corruption)

    def run(self):
        bootstrap_df = self.bootstrapper.run()
        record_df = run_model("Standard", bootstrap_df, cpu=True, batch_size=10)
        ground_truth = read_cifar10_ground_truth(os.path.join(self.source, "labels.txt"))
        self.conf, self.mu, self.sigma, self.satisfied = estimate_conf_int(record_df, self.rq_type, 1, ground_truth,
                                                                           0.95)
        self.done = True
