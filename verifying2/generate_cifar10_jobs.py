from tqdm import tqdm
from src.job import Cifar10Job
from src.utils import get_transformation_threshold, clean_dir
from src.constant import ROOT, CONTRAST, DEFOCUS_BLUR

DEFAULT_SOURCE = str(ROOT / 'data' / 'cifar10_pytorch' / 'val')
DEFAULT_DESTINATION = str(ROOT / 'bootstrap_data' / 'cifar10_pytorch')

cifar10_results = []
num_sample_iter = 2
sample_size = 5
jobs_queue_path = ROOT / 'jobs'
finished_job_path = ROOT / 'finished_jobs'
transformations = [CONTRAST, DEFOCUS_BLUR]
transformations = [CONTRAST]

clean_dir(jobs_queue_path)
clean_dir(finished_job_path)

model_names = ['Hendrycks2020AugMix_WRN',
               'Hendrycks2020AugMix_ResNeXt',
               'Kireev2021Effectiveness_Gauss50percent',
               'Kireev2021Effectiveness_RLATAugMixNoJSD',
               'Kireev2021Effectiveness_AugMixNoJSD',
               'Kireev2021Effectiveness_RLAT',
               'Standard']

counter = 0
pbar = tqdm(total=len(transformations) * 2)

for transformation in transformations:
    for rq_type in ["abs", "rel"]:
        threshold = get_transformation_threshold(transformation, rq_type)
        job = Cifar10Job(source=DEFAULT_SOURCE, destination=DEFAULT_DESTINATION, num_sample_iter=num_sample_iter,
                         sample_size=sample_size, transformation=transformation, rq_type=rq_type,
                         model_names=model_names, threshold=threshold, batch_size=5, cpu=True, bootstrapper=None)

        job.save(jobs_queue_path / f'{counter}.pickle')
        counter += 1
        pbar.update(n=1)
