import os
from typing import List

import pandas as pd
import pathlib2
from src.constant import CIFAR10C
from src.utils import load_cifar10_data, read_cifar10_ground_truth
from src.evaluate import run_model, estimate_conf_int, obtain_preserved_min_degradation
from .bootstrap import Cifar10CBootstrapper
from src.job import Job


class Cifar10CJob(Job):
    def __init__(self, source: str, destination: str, num_sample_iter: int, sample_size: int, corruption: str,
                 rq_type: str, cpu: bool, batch_size: int, model_names: List[str]):
        super(Cifar10CJob, self).__init__(source, destination, num_sample_iter, sample_size, cpu, batch_size,
                                          model_names)
        self.rq_type = rq_type
        self.corruption = corruption
        dataset_info_df = load_cifar10_data(pathlib2.Path(self.source))
        self.bootstrapper = Cifar10CBootstrapper(num_sample_iter=self.num_sample_iter, sample_size=self.sample_size,
                                                 source=source, destination=destination,
                                                 dataset_info_df=dataset_info_df, corruption=corruption)
        self.job_df = None

    def run(self):
        self.bootstrapper.run()
        results = []
        for model_name in self.model_names:
            record_df = run_model(model_name, self.bootstrapper.bootstrap_df, cpu=self.cpu, batch_size=self.batch_size)
            ground_truth = read_cifar10_ground_truth(os.path.join(self.source, "labels.csv"))
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
                "dataset": CIFAR10C,
                "num_sample_iter": self.num_sample_iter,
                "batch_size": self.batch_size,
                "corruption": self.corruption,
                "rq_type": self.rq_type,
                "model_name": model_name,
            })
        self.job_df = pd.DataFrame(data=results)
        self.done = True
        return self.job_df
