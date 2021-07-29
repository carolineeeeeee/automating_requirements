import os
import pickle
from typing import Dict
import pathlib2
from abc import ABC, abstractmethod
from src.bootstrap import Cifar10Bootstrapper
from src.constant import ROOT, GAUSSIAN_NOISE, IQA, IQA_PATH, matlabPyrToolsPath
from src.utils import start_matlab, load_cifar10_data, read_cifar10_ground_truth
from src.evaluate import run_model, estimate_conf_int


class Job(ABC):
    def __init__(self, source: str, destination: str, num_sample_iter: int, sample_size: int, cpu: bool, batch_size: int):
        self.source = source
        self.destination = destination
        self.num_sample_iter = num_sample_iter
        self.sample_size = sample_size
        self.done = False
        self.success = False
        self.cpu = cpu
        self.batch_size = batch_size

        self.conf = None
        self.mu = None
        self.sigma = None
        self.satisfied = None

    @abstractmethod
    def run(self):
        pass

    def __str__(self) -> str:
        return f"""
        source: {self.source}
        destination: {self.destination}
        num_sample_iter: {self.num_sample_iter}
        sample_size: {self.sample_size}
        done: {self.done}
        success: {self.success}
        conf: {self.conf}
        mu: {self.mu}
        sigma: {self.sigma}
        satisfied: {self.satisfied}
        """


class Cifar10Job(Job):
    def __init__(self, source: str, destination: str, num_sample_iter: int, sample_size: int, transformation: str,
                model_name: str, rq_type: str, batch_size: int=10, cpu: bool=True):
        super(Cifar10Job, self).__init__(source, destination, num_sample_iter, sample_size, cpu, batch_size)
        self.transformation = transformation
        self.model_name = model_name
        self.rq_type = rq_type
        dataset_info_df = load_cifar10_data(pathlib2.Path(self.source))
        self.bootstrapper = Cifar10Bootstrapper(num_sample_iter=self.num_sample_iter, sample_size=self.sample_size,
                                                source=self.source,
                                                destination=self.destination,
                                                threshold=0.95,
                                                dataset_info_df=dataset_info_df,
                                                transformation=transformation)

    def run(self):
        matlab_eng = start_matlab(IQA_PATH, matlabPyrToolsPath)
        bootstrap_df = self.bootstrapper.run(matlab_eng)
        record_df = run_model(self.model_name, bootstrap_df, cpu=self.cpu, batch_size=self.batch_size)
        ground_truth = read_cifar10_ground_truth(os.path.join(self.source, "labels.txt"))
        self.conf, self.mu, self.sigma, self.satisfied = estimate_conf_int(record_df, self.rq_type, 1, ground_truth,
                                                                           0.95)
        self.done = True
        self.success = True

    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'destination': self.destination,
            'num_sample_iter': self.num_sample_iter,
            'sample_size': self.sample_size,
            'done': self.done,
            'success': self.success,
            'model_name': self.model_name,
            'rq_type': self.rq_type,
            'transformation': self.transformation,
            'conf': self.conf,
            'mu': self.mu,
            'sigma': self.sigma,
            'satisfied': self.satisfied
        }


    def __str__(self) -> str:
        return f"""
        source: {self.source}
        destination: {self.destination}
        num_sample_iter: {self.num_sample_iter}
        sample_size: {self.sample_size}
        done: {self.done}
        success: {self.success}
        model_name: {self.model_name}
        rq_type: {self.rq_type}
        transformation: {self.transformation}
        conf: {self.conf}
        mu: {self.mu}
        sigma: {self.sigma}
        satisfied: {self.satisfied}
        """