from cifar10c.bootstrap import Cifar10CBootstrapper
from src.constant import ROOT
from src.utils import load_cifar10_data
source = ROOT / 'data' / 'cifar-10-c-images' / 'gaussian_blur'
destination = ROOT / 'bootstrap_data_c'
dataset_info_df = load_cifar10_data(source)
print(dataset_info_df)
b = Cifar10CBootstrapper(num_sample_iter=2, sample_size=10, source=source, destination=destination, dataset_info_df=dataset_info_df, corruption='gaussian_blur')
df = b.run()
print(df)