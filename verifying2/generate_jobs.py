from tqdm import tqdm

from src.utils import get_transformation_threshold, clean_dir
from src.constant import ROOT, ROBUSTBENCH_CIFAR10_MODEL_NAMES, CONTRAST_G, UNIFORM_NOISE, LOWPASS, HIGHPASS, \
    PHASE_NOISE, CONTRAST, DEFOCUS_BLUR, MOTION_BLUR, GLASS_BLUR, CIFAR10_C_CORRUPTION, IMAGENET_DATA_DIR
from src.job import Cifar10Job, ImagenetJob
from cifar10c.job import Cifar10CJob

DEFAULT_SOURCE = str(ROOT / 'data' / 'cifar10_pytorch' / 'val')
DEFAULT_DESTINATION = str(ROOT / 'bootstrap_data' / 'cifar10_pytorch')

cifar10_results = []
num_sample_iter = 2
sample_size = 10
jobs_queue_path = ROOT / 'jobs'
finished_job_path = ROOT / 'finished_jobs'
transformations = [CONTRAST, CONTRAST_G, UNIFORM_NOISE, LOWPASS, HIGHPASS, PHASE_NOISE, DEFOCUS_BLUR, MOTION_BLUR,
                   GLASS_BLUR]
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
        # cifar10 job
        job = Cifar10Job(source=DEFAULT_SOURCE, destination=DEFAULT_DESTINATION, num_sample_iter=num_sample_iter,
                         sample_size=sample_size, transformation=transformation, rq_type=rq_type,
                         model_names=model_names,
                         threshold=threshold, batch_size=5, cpu=False, bootstrapper=None)
        # imagenet job
        # job = ImagenetJob(source=IMAGENET_DATA_DIR / 'imgs', destination="./bootstrap_imagenet", num_sample_iter=num_sample_iter, 
        #                     sample_size=sample_size, transformation=transformation, rq_type=rq_type,
        #                     model_names=model_names, threshold=threshold, batch_size=5, cpu=True)


        job.save(jobs_queue_path / f'{counter}.pickle')
        counter += 1
        pbar.update(n=1)


# for corruption in CIFAR10_C_CORRUPTION:
#     for rq_type in ["abs", "rel"]:
#         # cifar10c job
#         job = Cifar10CJob(source=DEFAULT_SOURCE, destination=DEFAULT_DESTINATION, num_sample_iter=num_sample_iter, 
#                             sample_size=sample_size, corruption=corruption,
#                             rq_type=rq_type, cpu=True, batch_size=5, model_names=model_names)
#         job.save(jobs_queue_path / f'{counter}.pickle')
#         counter += 1
#         pbar.update(n=1)