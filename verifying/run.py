import argparse
from typing import List

from src.constant import ROOT, GAUSSIAN_NOISE, TRANSFORMATIONS, ROBUSTBENCH_CIFAR10_MODEL_NAMES
from src.job import Cifar10Job
from src.utils import get_transformation_threshold


def run(source: str, destination: str, num_sample_iter: int, sample_size: int, transformation: str,
        model_names: List[str], rq_type: str, batch_size: int, cpu: bool = True):
    threshold = get_transformation_threshold(transformation, rq_type)
    job = Cifar10Job(source, destination, num_sample_iter, sample_size,
                     transformation, rq_type, model_names, threshold, batch_size, cpu)
    job.run()
    return job


if __name__ == '__main__':
    DEFAULT_SOURCE = str(ROOT / 'data' / 'cifar10_pytorch' / 'val')
    DEFAULT_DESTINATION = str(ROOT / 'bootstrap_data' / 'cifar10_pytorch')
    parser = argparse.ArgumentParser("Run Cifar10")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="source of dataset")
    parser.add_argument("--destination", default=DEFAULT_DESTINATION, help="location to save bootstrapping images")
    parser.add_argument("--num_sample_iter", type=int, required=True, help="Number of bootstrap iterations")
    parser.add_argument("--sample_size", type=int, required=True,
                        help="Number of unique images per bootstrap iteration")
    parser.add_argument("--transformation", choices=TRANSFORMATIONS,
                        default=GAUSSIAN_NOISE, help="transformation to apply to images")
    parser.add_argument("--rq_type", choices=["abs", "rel"], required=True, help="requirement type")
    parser.add_argument("--model_names", nargs="+", choices=ROBUSTBENCH_CIFAR10_MODEL_NAMES, help="name of models")
    parser.add_argument("--batch_size", type=int, default=5, help="name of model to run")
    parser.add_argument(
        "--cpu", action="store_true", default=False,
        help="use CPU only while having GPU, will verify if GPU is available if this argument is set to False")
    args = parser.parse_args()
    run(args.source, args.destination, args.num_sample_iter, args.sample_size,
        args.transformation, args.model_names, args.rq_type, args.batch_size, args.cpu)
