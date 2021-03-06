.. _checking:

Requirement checking method
===========================

This automated method checks whether an MVC satisfies the instantiated reliability requirements.
Given a set with original images, our process generates test cases by transforming the original images within the range specified in the instantiated requirements, runs the tests on the model, and checks whether the MVC satisfies our requirements.

We provide in this page the implementation details.

Setup Procedures
----------------

Working Directory: ``automating_requirements/checking``

Author's OS and hardware:

-  OS: Ubuntu 20.04, architecture=amd64
-  RAM: 16GB
-  GPU: NVIDIA GTX

1. Recommended to use a virtual python environment with Anaconda,
   virtualenv works too but the rest of the instructions will be using
   Anaconda

    .. code-block:: bash
        :linenos:

        conda create -n envname python=3.7
        conda activate envname

2. Install PyTorch (with cuda if you have Nvidia GPU)

    Check https://pytorch.org/get-started/locally/ for the installation
    commands

    For me, I have cuda 11.1, and run

    .. code-block:: bash
        :linenos:

        conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

3. Install RobustBench

    https://robustbench.github.io/

    https://github.com/RobustBench/robustbench

    .. code-block:: bash
        :linenos:

        pip install git+https://github.com/RobustBench/robustbench.git@v0.2.1


4. Install matlab (I used 2021a without any addition functionalities)
5. Install matlab package for Python following
   https://www.mathworks.com/help/matlab/matlab\_external/install-the-matlab-engine-for-python.html

    If you are using Anaconda for virtual environment, running
    ``python setup.py install`` will install matlab package for your default
    system python, not for your virtual environment.

    **Solution:** run ``which python`` while your virtual environment is
    activated and get your virtual environment python executable path, then
    run ``setup.py`` with this executable.


    .. code-block:: bash
        :linenos:
        
        conda activate envname
        conda_python=$(which python)
        sudo $conda_python setup.py install

6. A ``pythonenv.yml`` and a ``requirements.txt`` are provided just in
   case some packages are missing. Refer to these files if some packages
   are missing while running the code.

    ``pythonenv.yml`` describes the entire conda environment on the author's
    machine

    ``requirements.txt`` listed a few python packages that may not be
    installed by previous steps. Run ``pip install requirements.txt`` to
    install them.

    .. code-block:: bash
        :linenos:

        pip install -r requirements.txt

7. Run ``apt-get install libmagickwand-dev``

8. Download ``image-quality-tools`` and place it under ``utils`` directory as ``utils/image-quality-tools``.

9. Run ``make all`` within ``automating_requirements/checking``

   Breakdown: ``make all`` will run the following commands which you can run them individually:

    See the Makefile comments for more detailed explanation

    -  ``make download_cifar10``

    Download cifar10 dataset from
    https://www.cs.toronto.edu/~kriz/cifar.html using pytorch

    -  ``make download_cifar10_c``

    Download cifar10-c dataset from https://zenodo.org/record/2535967,
    this dataset may take longer to download

    -  ``make download_imagenet_mapping``

    -  download MSCOCO\_to\_ImageNet\_category\_mapping.txt: map imagenet
        basic class to labels ids

    -  download synset\_words.txt: map imagenet label ids to english
        words

    -  ``make download_imagenet_bbox``

    -  download imagenet validation bounding boxes containing image
        labels from
        https://image-net.org/data/ILSVRC/2012/ILSVRC2012\_bbox\_val\_v3.tgz

    -  ``make download_imagenet_val_img``

    -  download imagenet validation images from
        https://image-net.org/data/ILSVRC/2012/ILSVRC2012\_img\_val.tar

    -  ``make produce_imagenet_label``

    -  download imagenet\_label.sh file for labelling images
    -  run ``imagenet_label.sh``, this depends on the files
        ``./data/imagenet/val/*.xml``, so ``make download_imagenet_bbox``
        is a dependency, which generates ``val`` directory
    -  Note: if your network isn't good, you may fail to download
        ``imagenet_label.sh``, an empty file could be saved to your file
        system, no error will be raised but the rest of the code will
        fail. So if something went wrong here, check if this file is
        empty.
    -  run ``./prepare/prepare_imagenet.py`` to produce all kinds of
        imagenet label mapping in json and csv format for easier access in
        the future, these files are saved in ``./data/imagenet/info``

    -  ``make preprocess_cifar10_pytorch``

    Preprocess the downloaded dataset to save the images in the desired
    sturcture

    -  ``make preprocess_cifar10c``

    Preprocess the downloaded dataset to save the images in the desired
    sturcture

How to run?
-----------

Run a single experiment on cifar10 dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``run.py`` is responsible for running an experiment.

Run ``python run.py --help`` to see all the options.

Sample Command
^^^^^^^^^^^^^^

.. code-block:: bash
    :linenos:

    python run_cifar10.py --num_sample_iter 2 --sample_size 10 --transformation gaussian_noise --rq_type correction_preservation --model_names Standard Hendrycks2020AugMix_ResNeXt --batch_size 5

Run a single experiment on ImageNet dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to ``run_cifar10.py``

Run Multiple Experiments
~~~~~~~~~~~~~~~~~~~~~~~~

Running multiple jobs is risky, because the jobs could terminate in the
middle due to various reasons such as lack of GPU memory, bugs, or
incorrect setup. Then you may lose your progress, and be forced to
restart all experiments.

Thus, we designed each experiment to be a Python Job object which can be
saved with pickle. Jobs in the queue will be saved in ``jobs``
directory, all finished jobs will be moved to ``finished_jobs``
directory.

The configuration and experiment results for each job will also be saved
in the finished jobs objects. You can use a script the read these
objects and parse the results.

See ``run-all.py`` for an example about how to generate the jobs
objects. You can tweak the parameters at the top of the file for
creating jobs.

See ``parse_results.py`` for example about how to read the job results.

Clean Up
--------

Running this project requirements lots of space for storing images, run
``make clean`` to free up spaces.

Or if something went wrong while downloading the required files and you
couldn't fix it, you can run ``make clean`` and restart (downloading
everything again takes a long time).

File Responsibility
-------------------

In this seciton, I will describe what the files in ``checking`` folder
is responsible for.

-  ``Makefile``: convenience script/commands for downloading and preprocessing datasets
-  ``data/``: where datasets and their related files will be stored
-  ``prepare/``: script for preprocessing downloaded datasets and turn them into desired format
-  ``prepare_imagenet.py``
-  ``preprocess_cifar10_c.py``
-  ``preprocess_cifar10_pytorch.py``
-  ``src/``: main source files, containing all logics of this project
-  ``bootstrap.py``: bootstrapping logic
-  ``constant.py``: constant variables
-  ``dataset.py``: definition or PyTorch Dataset models for data loading
-  ``evaluate.py``
-  ``job.py``: Definition of job objects, each experiment is a job
-  ``utils.py``: All utility functions
-  ``Imagenet_c_transformations.py``: definitions of all image transformations
-  ``utils/``
-  ``clean.sh``: a shell script for cleaning up data directory (removing all datasets and their related files)
-  ``image-quality-tools``: a directory user needs to download place here manually, containing matlab code for calculating IQA scores
-  ``generate_*``
-  ``generate_cifar10_jobs.py``: generate jobs files on cifar10 dataset and save into **jobs** directory
-  ``generate_imagenet_jobs.py``: generate jobs files on imagenet dataset and save into **jobs** directory
-  ``run_cifar10.py``: run a single Cifar10 job
-  ``run_imagenet.py``: run a single ImageNet job
-  ``run_jobs.py``: run all jobs saved in **jobs** directory and move the successful ones into **finished\_jobs** directory
-  ``parse_jobs_results.py``: parse experiment results from job files in **finished\_jobs** directory
-  ``generate_transformed_images.py``: generate a transformed image dataset from existing dataset given a transformation type
-  ``finetune.ipynb``: finetuning experiment on transformed image dataset
