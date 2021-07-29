from src.bootstrap import Cifar10Bootstrapper
from src.constant import ROOT, GAUSSIAN_NOISE, IQA, IQA_PATH, matlabPyrToolsPath
from src.utils import start_matlab, load_cifar10_data
from src.evaluate import run_model, estimate_conf_int
from src.job import Cifar10Job
from tabulate import tabulate
from cifar10c.job import Cifar10CJob
import pickle

if __name__ == '__main__':
    source = ROOT / 'data' / 'cifar10_pytorch' / 'val'
    destination = ROOT / 'bootstrap_data' / 'cifar10_pytorch'
    eng = start_matlab()
    job = Cifar10Job(source, destination, num_sample_iter=2, sample_size=10, transformation=GAUSSIAN_NOISE,
                     model_name='Standard', rq_type='abs')
    # f = open('./tmp.pickle', 'wb')
    # pickle.dump(job, f)
    # f.close()

    # f = open('./tmp.pickle', 'rb')
    # job = pickle.load(f)
    # print(job)


    # result = job.run()
    # print(result)

    # source = ROOT / 'data' / 'cifar-10-c-images' / 'gaussian_blur'
    # destination = ROOT / 'bootstrap_data_c'
    # job = Cifar10CJob(source, destination, 2, 10, 'gaussian_blur', 'Standard', 'abs')
    job.run(eng)