import os
import pickle

import pandas as pd
import pathlib2
from typing import Dict, Union, List
from abc import ABC, abstractmethod

from src.bootstrap import Cifar10Bootstrapper, Bootstrapper
from src.constant import ROOT, GAUSSIAN_NOISE, IQA, IQA_PATH, matlabPyrToolsPath, CIFAR10
from src.utils import start_matlab, load_cifar10_data, read_cifar10_ground_truth, dict_to_str
from src.evaluate import run_model, estimate_conf_int, obtain_preserved_min_degradation


class Job(ABC):
    def __init__(self, source: str, destination: str, num_sample_iter: int, sample_size: int, cpu: bool,
                 batch_size: int, model_names: List[str]):
        self.source = source
        self.destination = destination
        self.num_sample_iter = num_sample_iter
        self.sample_size = sample_size
        self.done = False
        self.cpu = cpu
        self.batch_size = batch_size
        self.model_names = model_names

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
        model names: {self.model_names}
        """


class Cifar10Job(Job):
    def __init__(
            self, source: str, destination: str, num_sample_iter: int, sample_size: int, transformation: str,
            rq_type: str, model_names: List[str], threshold: float = 0.95,
            batch_size: int = 10, cpu: bool = True, bootstrapper: Cifar10Bootstrapper = None):
        super(Cifar10Job, self).__init__(source, destination, num_sample_iter, sample_size, cpu, batch_size,
                                         model_names)
        self.transformation = transformation
        self.rq_type = rq_type
        self.threshold = threshold
        self.dataset_info_df = load_cifar10_data(pathlib2.Path(self.source))
        self.bootstrapper = self.gen_bootstrapper() if bootstrapper is None else bootstrapper
        self.job_df = None

    def gen_bootstrapper(self) -> Cifar10Bootstrapper:
        return Cifar10Bootstrapper(num_sample_iter=self.num_sample_iter, sample_size=self.sample_size,
                                   source=self.source,
                                   destination=self.destination,
                                   threshold=self.threshold,
                                   dataset_info_df=self.dataset_info_df,
                                   transformation=self.transformation)

    def run(self) -> pd.DataFrame:
        """[summary]

        :param bootstrapper: optional bootstrapper object, in case you need to reuse a bootstrapper object and images for multiple jobs
        :type bootstrapper: Union[Bootstrapper, None]
        :raises ValueError: [description]
        """
        matlab_eng = start_matlab(IQA_PATH, matlabPyrToolsPath)
        self.bootstrapper.run(matlab_eng)
        results = []
        for model_name in self.model_names:
            record_df = run_model(model_name, self.bootstrapper.bootstrap_df, cpu=self.cpu, batch_size=self.batch_size)
            ground_truth = read_cifar10_ground_truth(os.path.join(self.source, "labels.txt"))
            if self.rq_type == 'rel':
                a = obtain_preserved_min_degradation(record_df)
                conf, mu, sigma, satisfied = estimate_conf_int(record_df, self.rq_type, 1, ground_truth, a)
            elif self.rq_type == 'abs':
                conf, mu, sigma, satisfied = estimate_conf_int(record_df, self.rq_type, 1, ground_truth, 0.95)
            else:
                raise ValueError("Invalid rq_type")
            results.append({
                "conf": conf,
                "mu": mu,
                "sigma": sigma,
                "satisfied": satisfied,
                "dataset": CIFAR10,
                "num_sample_iter": self.num_sample_iter,
                "threshold": self.threshold,
                "batch_size": self.batch_size,
                "transformation": self.transformation,
                "rq_type": self.rq_type,
                "model_name": model_name,
            })
        self.job_df = pd.DataFrame(data=results)
        self.done = True
        return self.job_df

    def save(self, path: str) -> None:
        f = open(path, 'wb')
        pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        f = open(path, 'rb')
        return pickle.load(f)

    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'destination': self.destination,
            'num_sample_iter': self.num_sample_iter,
            'sample_size': self.sample_size,
            'done': self.done,
            'model_names': self.model_names,
            'rq_type': self.rq_type,
            'transformation': self.transformation,
        }

    def __str__(self) -> str:
        return dict_to_str(self.to_dict())
