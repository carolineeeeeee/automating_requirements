# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from run import run
import pandas as pd
from src.constant import ROBUSTBENCH_CIFAR10_MODEL_NAMES, CONTRAST_G, UNIFORM_NOISE, LOWPASS, HIGHPASS, PHASE_NOISE, CONTRAST
from src.helper import get_transformation_threshold


# %%
cifar10_results = []
num_batch = 200
batch_size = 200
for model_name in ROBUSTBENCH_CIFAR10_MODEL_NAMES[:3]:
    for transformation in [CONTRAST, CONTRAST_G, UNIFORM_NOISE, LOWPASS, HIGHPASS, PHASE_NOISE]:
        for rq_type in ["abs", "rel"]:
            threshold = get_transformation_threshold(transformation, rq_type)
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
            conf, mu, sigma, satisfied = run(num_batch, batch_size, transformation,
                                             "cifar10_data/val", "./bootstrap_output", rq_type, model_name, threshold)
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
