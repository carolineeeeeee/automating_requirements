import pickle
import shutil

from src.job import Cifar10Job
from src import constant


if __name__ == '__main__':
    jobs_queue_path = constant.ROOT / 'jobs'
    finished_job_path = constant.ROOT / 'finished_jobs'

    for job_file in jobs_queue_path.iterdir():
        job = Cifar10Job.load(str(job_file))
        # try:
        job.run()
        job.save(str(job_file))
        shutil.move(job_file, finished_job_path/job_file.name)
        # except Exception as e:
        #     print("error..")
        #     print(e)
