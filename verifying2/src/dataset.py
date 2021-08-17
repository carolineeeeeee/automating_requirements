import torch
import pathlib2
import pandas as pd
from PIL import Image
from typing import Dict
from torchvision import transforms
from torch.utils.data import Dataset

ToTensor = transforms.ToTensor()


class Cifar10Dataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index()
        self.data = self.df.to_dict('records')

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        data = self.data[index]
        original_image = Image.open(data['original_path'])
        transformed_image = Image.open(data['new_path'])
        data.update({
            'original_image': ToTensor(original_image),
            'new_image': ToTensor(transformed_image),
        })
        return data


class GeneralDataset(Dataset):
    def __init__(self, dataset_path: pathlib2.Path):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(str(self.dataset_path / 'labels.csv'), index_col=0)
        self.df['filepath'] = self.df['filename'].apply(lambda filename: str(self.dataset_path / filename))
        self.data = self.df.to_dict('records')

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        data = self.data[index]
        image = Image.open(data['filepath'])
        label = data['label']
        return ToTensor(image), label
