from tqdm import tqdm
from src.utils import get_transformation_threshold, clean_dir
from src.constant import ROOT, CONTRAST, DEFOCUS_BLUR, IMAGENET_DATA_DIR, IMAGENET_MODELS, \
    CORRECTION_PRESERVATION, PREDICTION_PRESERVATION
from src.job import ImagenetJob

DEFAULT_SOURCE = str(IMAGENET_DATA_DIR / 'imgs')
DEFAULT_DESTINATION = str(ROOT / 'bootstrap_imagenet')

cifar10_results = []
num_sample_iter = 100
sample_size = 100
jobs_queue_path = ROOT / 'jobs'
finished_job_path = ROOT / 'finished_jobs'
# transformations = [CONTRAST, DEFOCUS_BLUR]
transformations = [CONTRAST]

clean_dir(jobs_queue_path)
clean_dir(finished_job_path)

model_names = IMAGENET_MODELS
counter = 0
pbar = tqdm(total=len(transformations) * 2)

for i in range(2):
    threshold = get_transformation_threshold(CONTRAST, CORRECTION_PRESERVATION)
    job = ImagenetJob(source=DEFAULT_SOURCE, destination=DEFAULT_DESTINATION,
                      num_sample_iter=num_sample_iter,
                      sample_size=sample_size, transformation=CONTRAST, rq_type=CORRECTION_PRESERVATION,
                      model_names=model_names, threshold=threshold, batch_size=5, cpu=False, random_seed=counter)
    job.save(jobs_queue_path / f'{counter}.pickle')
    counter += 1

# for transformation in transformations:
#     for rq_type in [ACCURACY_PRESERVATION, PREDICTION_PRESERVATION]:
#         threshold = get_transformation_threshold(transformation, rq_type)
#         job = ImagenetJob(source=DEFAULT_SOURCE, destination=DEFAULT_DESTINATION,
#                           num_sample_iter=num_sample_iter,
#                           sample_size=sample_size, transformation=transformation, rq_type=rq_type,
#                           model_names=model_names, threshold=threshold, batch_size=5, cpu=True)
#         job.save(jobs_queue_path / f'{counter}.pickle')
#         counter += 1
#         pbar.update(n=1)
