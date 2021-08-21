import random
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
        pil_img = Image.open(data['filepath'])
        image = ToTensor(pil_img)
        pil_img.close()
        label = data['label']
        IQA_score = data['IQA_score'] if 'IQA_score' in data else 1
        return image, label, IQA_score, data['filepath']


class GeneralDataset2(Dataset):
    def __init__(self, dataset_path: pathlib2.Path, alternative_dataset_path: pathlib2.Path,
                 alternative_prob: float = 0.5):
        self.datapaths = {'main': dataset_path, 'alt': alternative_dataset_path}
        self.alternative_prob = alternative_prob
        self.info_df = {
            type_: pd.read_csv(str(self.datapaths[type_] / 'labels.csv'),
                               index_col=0) for type_ in self.datapaths.keys()}
        for type_ in self.datapaths.keys():
            self.info_df[type_]['filepath'] = self.info_df[type_]['filename'].apply(
                lambda filename: str(self.dataset_path / filename))
        # self.data = {}
        self.data = self.df.to_dict('records')
        self.alternative_data = self.alternative_df.to_dict('records')

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict:
        data = self.alternative_data[index] if random.random() < self.alternative_prob else self.data[index]
        pil_img = Image.open(data['filepath'])
        image = ToTensor(pil_img)
        pil_img.close()
        label = data['label']
        if 'IQA_score' in data:
            IQA_score = data['IQA_score']
        else:
            IQA_score = 0
        return image, label, IQA_score, data['filepath']
