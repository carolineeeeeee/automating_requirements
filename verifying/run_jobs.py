import shutil
import argparse
from src.job import Cifar10Job, ImagenetJob
from src.constant import CIFAR10, IMAGENET
from src import constant


if __name__ == '__main__':
    parser = argparse.ArgumentParser("run job parser")
    parser.add_argument("-d", "--dataset", choices=[CIFAR10, IMAGENET],
                        default=CIFAR10, help="Class/type of the job files")
    args = parser.parse_args()
    if args.dataset == CIFAR10:
        job_type = Cifar10Job
    elif args.dataset == IMAGENET:
        job_type = ImagenetJob
    else:
        raise ValueError("Invalid dataset: %s" % args.dataset)
    jobs_queue_path = constant.ROOT / 'jobs'
    finished_job_path = constant.ROOT / 'finished_jobs'

    for job_file in jobs_queue_path.iterdir():
        print(f"Running file {str(job_file)}")
        job = job_type.load(str(job_file))
        # try:
        job.run()
        job.save(str(job_file))
        shutil.move(job_file, finished_job_path/job_file.name)
        # except Exception as e:
        #     print("error..")
        #     print(e)
