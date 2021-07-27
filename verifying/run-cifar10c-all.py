# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from run import run
import pandas as pd
from src.constant import ROBUSTBENCH_CIFAR10_MODEL_NAMES
from run_cifar10c import run, preparation_and_bootstrap


# %%
cifar10_c_results = []
num_batch = 100
batch_size = 100

for transformation in [
        "brightness", "contrast", "fog", "frost", "gaussian_blur", "gaussian_noise", "jpeg_compression", "snow"]:
    for rq_type in ["abs", "rel"]:
        # bootstrap
        ground_truth, image_df = preparation_and_bootstrap(num_batch, batch_size, "./cifar10_c_data",
                                             "./cifar10c_bootstrap_output", transformation)
        for model_name in ROBUSTBENCH_CIFAR10_MODEL_NAMES[:3]:
            record = {
                "num_batch": num_batch,
                "batch_size": batch_size,
                "transformation": transformation,
                "rq_type": rq_type,
                "model_name": model_name,
            }
            print(record)
            # try:
            conf, mu, sigma, satisfied = run(ground_truth, image_df, rq_type, model_name)
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
            cifar10_c_results.append(record)
            df = pd.DataFrame(data=cifar10_c_results)
            df.to_csv("result-cifar10c.csv")


# %%


# %%


# %%


# %%


# %%


# %%


# %%