**********
Estimating
**********

As promised, here's the scripts for estimating and the two files containing human experiment results. For the files in estimating, user just need to run parse_exp_results.py first then run estimate_range.R
A fews that we need to be careful about:
parse_exp_results.py, since it needs the two experiment result csv files, depending on the directory we end up putting the result csv files,  the path on line 22 and 41 need to be updated
in estimate_range.R, it reads files that are created by parse_exp_results.py. So there are paths that have my name on it,  this also need to be updated for the final version.
We also allow people to re-create our results in the paper, thus I have the folders cifar10, imagenet and sixty percent. We just need to add instructions for that.

Check ``estimating`` directory for the code.

Check ``cifar10_experiment_results.csv`` and ``imagenet_experiment_results.csv`` for the results.