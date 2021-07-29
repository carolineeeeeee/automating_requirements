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
