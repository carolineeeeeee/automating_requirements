import os
import sys
import torch
import shutil
import logging
import pathlib2
import torchvision
import pandas as pd
from tqdm import tqdm
from typing import Union
import torch.multiprocessing

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(filename)s | %(lineno)d | %(message)s',
                              '%m-%d-%Y %H:%M:%S')
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

torch.multiprocessing.set_sharing_strategy('file_system')

__dir__ = pathlib2.Path(__file__).absolute().parent
data_dir = __dir__.parent / 'data'
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)
toPIL = torchvision.transforms.ToPILImage()


def save(dataset_root: Union[pathlib2.Path, str], output_dir: Union[pathlib2.Path, str], train: bool):
    output_dir = pathlib2.Path(output_dir).absolute()
    dataset_root = pathlib2.Path(dataset_root).absolute()
    logger.info(f"Remove target output directory {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames, labels = [], []
    dataset = torchvision.datasets.CIFAR10(root=dataset_root, train=train, download=True,
                                           transform=torchvision.transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    logger.info(f"Saving images to {output_dir}")
    pbar = tqdm(total=len(dataset))
    for i, data in enumerate(dataloader):
        for j in range(len(data[0])):
            img = toPIL(data[0][j])
            filename = f'{i}.png'
            img.save(os.path.join(output_dir, filename))
            img.close()
            labels.append(int(data[1][0]))
            filenames.append(filename)
            pbar.update()
    label_path = output_dir / 'labels.csv'
    label_df = pd.DataFrame(data={
        'filename': filenames,
        'label': labels
    })
    label_df.to_csv(label_path)
    logger.info(f"Saving labelex. to {label_path}")


if __name__ == "__main__":
    save(data_dir, data_dir / "cifar10_pytorch" / "val", False)
    save(data_dir, data_dir / "cifar10_pytorch" / "train", True)
    logger.info("Finished")
