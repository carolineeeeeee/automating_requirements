import argparse

from src.constant import ROOT, GAUSSIAN_NOISE, TRANSFORMATIONS
from cifar10c.job import Cifar10CJob


def run(source: str, destination: str, num_sample_iter: int, sample_size: int, model_name: str, corruption: str,
        rq_type: str, batch_size: int, cpu: bool = True):
    job = Cifar10CJob(source, destination, num_sample_iter, sample_size, corruption,
                 rq_type, cpu, batch_size, model_names=[model_name])
    job.run()
    return job


if __name__ == '__main__':
    DEFAULT_SOURCE = str(ROOT / 'data' / 'cifar-10-c-images' / 'gaussian_blur')
    DEFAULT_DESTINATION = str(ROOT / 'bootstrap_data_c')
    parser = argparse.ArgumentParser("Run Cifar10 C")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="source of dataset")
    parser.add_argument("--destination", default=DEFAULT_DESTINATION, help="location to save bootstrapping images")
    parser.add_argument("--num_sample_iter", required=True, type=int, help="Number of bootstrap iterations")
    parser.add_argument("--sample_size", required=True, type=int, help="Number of unique images per bootstrap iteration")
    parser.add_argument("--corruption", choices=TRANSFORMATIONS,
                        default=GAUSSIAN_NOISE, help="corruption to apply to images")
    parser.add_argument("--rq_type", choices=["abs", "rel"], required=True, help="requirement type")
    parser.add_argument("--model_name", required=True, type=str, help="name of model to run")
    parser.add_argument("--batch_size", type=int, default=5, help="name of model to run")
    parser.add_argument(
        "--cpu", action="store_true", default=False,
        help="use CPU only while having GPU, will verify if GPU is available if this argument is set to False")
    args = parser.parse_args()
    run(args.source, args.destination, args.num_sample_iter, args.sample_size, args.model_name,
        args.corruption, args.rq_type, args.batch_size, args.cpu)
