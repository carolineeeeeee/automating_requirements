from tabulate import tabulate
import pandas as pd
from src.constant import ROOT
from src.job import Cifar10Job

finished_job_path = ROOT / 'finished_jobs'

results = []
for path in finished_job_path.iterdir():
    job = Cifar10Job.load(str(path))
    results.append(job.to_dict())
results_df = pd.DataFrame(data=results)
print(tabulate(results_df, headers='keys', tablefmt='pretty'))
print("Parsed results saved to results.csv")
results_df.to_csv("results.csv")
