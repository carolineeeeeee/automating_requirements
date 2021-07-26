import os
import torch
import shutil
import pathlib2
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


__root__ = pathlib2.Path(__file__).absolute().parent
static_dir = __root__/'static'
if not (static_dir).exists():
    static_dir.mkdir(parents=True, exist_ok=True)
toPIL = torchvision.transforms.ToPILImage()


def save(output_dir: str, train: bool):
    output_dir = pathlib2.Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    dataset = torchvision.datasets.CIFAR10(root="./", train=train, download=True,
                                           transform=torchvision.transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    for i, data in enumerate(tqdm(dataloader)):
        img = toPIL(data[0][0])
        img.save(os.path.join(output_dir, f'{i}.png'))
        img.close()
        labels.append(data[1][0])
    np.savetxt(os.path.join(output_dir, f'labels.txt'), np.asarray(labels))


if __name__ == "__main__":
    save("./cifar10_data/val", False)
    save("./cifar10_data/train", True)
