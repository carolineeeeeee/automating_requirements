# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.insert(1, '/u/boyue/ICSE2022-resubmission/lib')
from run import run, preparation_and_bootstrap
import pandas as pd
from src.constant import ROBUSTBENCH_CIFAR10_MODEL_NAMES, CONTRAST_G, UNIFORM_NOISE, LOWPASS, HIGHPASS, PHASE_NOISE, CONTRAST, DEFOCUS_BLUR, MOTION_BLUR, GLASS_BLUR
from src.helper import get_transformation_threshold


# %%
cifar10_results = []
num_batch = 200
batch_size = 100

for transformation in [CONTRAST, CONTRAST_G, UNIFORM_NOISE, LOWPASS, HIGHPASS, PHASE_NOISE, DEFOCUS_BLUR, MOTION_BLUR, GLASS_BLUR]:
    for rq_type in ["abs", "rel"]:
        # bootstrap
        threshold = get_transformation_threshold(transformation, rq_type)
        ground_truth, boot_df = preparation_and_bootstrap(num_batch, batch_size,
                                             "cifar10_data/val", "./bootstrap_output", rq_type, transformation, threshold)
        for model_name in ROBUSTBENCH_CIFAR10_MODEL_NAMES:
            record = {
                "dataset": "cifar10",
                "num_batch": num_batch,
                "threshold": threshold,
                "batch_size": batch_size,
                "transformation": transformation,
                "rq_type": rq_type,
                "model_name": model_name,
            }
            print(record)
            # try:
            conf, mu, sigma, satisfied = run(ground_truth, boot_df, rq_type, model_name)
            record.update({
                "conf": conf,
                "mu": mu,
                "sigma": sigma,
                "satisfied": satisfied,
                "success": True
            })
            # except Exception as e:
            #     print(e)
            #     record.update({
            #         "conf": 0,
            #         "mu": 0,
            #         "sigma": 0,
            #         "satisfied": 0,
            #         "success": False
            #     })
            cifar10_results.append(record)
            df = pd.DataFrame(data=cifar10_results)
            df.to_csv("result-cifar-1.csv")


# %%
df = pd.DataFrame(data=cifar10_results)
df


# %%
