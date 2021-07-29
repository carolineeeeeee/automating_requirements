# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import numpy as np
from typing import Dict
import multiprocessing as mp
from torchvision import transforms
from torch.utils.data import Dataset
from src.utils import get_model


# %%
ToTensor = transforms.ToTensor()


# %%

class Cifar10CDataset(Dataset):
    def __init__(self, file: str, label_file: str):
        self.data = np.load(file)
        self.labels = np.load(label_file)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict:
        return ToTensor(self.data[index]), self.labels[index]


# %%
dataset = Cifar10CDataset("./data/CIFAR-10-C/contrast.npy", "./data/CIFAR-10-C/labels.npy")


# %%
dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False, num_workers=mp.cpu_count())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# %%
model = get_model('Standard').to(device)


# %%
for i, (images, labels) in enumerate(dataloader):
    images = images.to(device)
    labels = labels.to(device)
    predictions = model(images)
    print(predictions)
    break


# %%



