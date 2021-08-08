# Verifying

## Setup Procedures

Working Directory: `automating_requirements/verifying`

1. Run `pip install requirements.txt`
2. Download `image-quality-tools` and place it under `utils` directory
3. Run `make all` within `automating_requirements/verifying`
    
    Breakdown: `make all` will run the following commands which you can run them individually:
    1. `make download_cifar10`
       
        download cifar10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html using pytorch
    2. `make download_cifar10_c`
       
        download cifar10-c dataset from https://zenodo.org/record/2535967, this dataset may take longer to download
    3. `make preprocess_cifar10_pytorch`
        
        Preprocess the downloaded dataset to save the images in the desired sturcture
    4. `make preprocess_cifar10c`
    
        Preprocess the downloaded dataset to save the images in the desired sturcture


## How to run?

### Run a single experiment on cifar10 dataset
`run.py` is responsible for running an experiment.

Run `python run.py --help` to see all the options.

#### Sample Command
```bash
python run.py --num_sample_iter 2 --sample_size 10 --transformation gaussian_noise --rq_type abs --model_names Standard  Hendrycks2020AugMix_ResNeXt --batch_size 5
```

### Run a single experiment on cifar10-c dataset

Similar to `run.py`

### Run Multiple Experiments

Running multiple jobs is risky, because the jobs could terminate in the middle due to various reasons such as lack of 
GPU memory, bugs, or incorrect setup. Then you may lose your progress, and be forced to restart all experiments.

Thus, we designed each experiment to be a Python Job object which can be saved with pickle.
Jobs in the queue will be saved in `jobs` directory, all finished jobs will be moved to `finished_jobs` directory.

The configuration and experiment results for each job will also be saved in the finished jobs objects. 
You can use a script the read these objects and parse the results.

See `run-all.py` for an example about how to generate the jobs objects. 
You can tweak the parameters at the top of the file for creating jobs.

See `parse_results.py` for example about how to read the job results.




## Clean Up

Running this project requirements lots of space for storing images, run `make clean` to free up spaces.
