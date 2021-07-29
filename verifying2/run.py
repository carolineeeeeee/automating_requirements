import os

import pathlib2

from src.bootstrap import Cifar10Bootstrapper
from src.constant import ROOT, GAUSSIAN_NOISE, IQA, IQA_PATH, matlabPyrToolsPath
from src.utils import start_matlab, load_cifar10_data, read_cifar10_ground_truth
from src.evaluate import run_model, estimate_conf_int
from tabulate import tabulate
from typing import Union


def run(source: str, destination: str, num_sample_iter: int, sample_size: int, transformation: str, rq_type: str):
    matlab_eng = start_matlab(IQA_PATH, matlabPyrToolsPath)
    dataset_info_df = load_cifar10_data(pathlib2.Path(source))
    bootstrapper = Cifar10Bootstrapper(num_sample_iter=num_sample_iter, sample_size=sample_size, source=source,
                                       destination=destination,
                                       threshold=0.95, dataset_info_df=dataset_info_df,
                                       transformation=transformation)
    df = bootstrapper.run(matlab_eng)
    record_df = run_model("Standard", df, cpu=True, batch_size=10)
    ground_truth = read_cifar10_ground_truth(os.path.join(source, "labels.txt"))
    conf, mu, sigma, satisfied = estimate_conf_int(record_df, rq_type, 1, ground_truth, 0.95)
    print(conf, mu, sigma, satisfied)
source = ROOT / 'data' / 'cifar10_pytorch' / 'val'
destination = ROOT / 'bootstrap_data' / 'cifar10_pytorch'
run(source, destination, 2, 10, GAUSSIAN_NOISE, 'abs')



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser("Run Cifar10 Bootstrapping")
#     parser.add_argument("--num_batch", type=int, default=10, help="Number of bootstrap iterations")
#     parser.add_argument("--batch_size", type=int, default=10, help="Number of unique images per bootstrap iteration")
#     parser.add_argument("--transformation", type=str, required=True,
#                         choices=TRANSFORMATIONS, help="Transformation to use")
#     parser.add_argument("--data_path", type=str, default=os.path.join(ROOT_PATH, "cifar10_data", "val"),
#                         help="Path to dataset, should be a folder containing images and a labels.txt")
#     parser.add_argument("--gen_path", type=str, default=os.path.join(ROOT_PATH, "bootstrap_output"),
#                         help="location to save intermediate bootstrapping images")
#     parser.add_argument("--rq_type", required=True, choices=['abs', 'rel'], help="requirement type")
#     parser.add_argument("--model_name", required=True, choices=ROBUSTBENCH_CIFAR10_MODEL_NAMES, help="Model Name")
#
#     args = parser.parse_args()
#     run(args.num_batch, args.batch_size, args.transformation, args.data_path, args.gen_path, args.rq_type,
#         args.model_name)
