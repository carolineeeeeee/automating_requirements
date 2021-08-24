from tqdm import tqdm
from src.utils import get_transformation_threshold, clean_dir
from src.constant import ROOT, CONTRAST_G, UNIFORM_NOISE, LOWPASS, HIGHPASS, \
    PHASE_NOISE, CONTRAST, DEFOCUS_BLUR, MOTION_BLUR, GLASS_BLUR, IMAGENET_DATA_DIR, IMAGENET_MODELS
from src.job import ImagenetJob

DEFAULT_SOURCE = str(IMAGENET_DATA_DIR / 'imgs')
DEFAULT_DESTINATION = str(ROOT / 'bootstrap_imagenet')

cifar10_results = []
num_sample_iter = 2
sample_size = 5
jobs_queue_path = ROOT / 'jobs'
finished_job_path = ROOT / 'finished_jobs'
transformations = [CONTRAST, CONTRAST_G, UNIFORM_NOISE, LOWPASS, HIGHPASS, PHASE_NOISE, DEFOCUS_BLUR, MOTION_BLUR,
                   GLASS_BLUR]
transformations = [CONTRAST]

clean_dir(jobs_queue_path)
clean_dir(finished_job_path)

model_names = IMAGENET_MODELS
counter = 0
pbar = tqdm(total=len(transformations) * 2)

for transformation in transformations:
    for rq_type in ["abs", "rel"]:
        threshold = get_transformation_threshold(transformation, rq_type)
        job = ImagenetJob(source=DEFAULT_SOURCE, destination=DEFAULT_DESTINATION,
                          num_sample_iter=num_sample_iter,
                          sample_size=sample_size, transformation=transformation, rq_type=rq_type,
                          model_names=model_names, threshold=threshold, batch_size=5, cpu=True)
        job.save(jobs_queue_path / f'{counter}.pickle')
        counter += 1
        pbar.update(n=1)
