import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm

toPIL = torchvision.transforms.ToPILImage()


def save(output_dir: str, train: bool):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    labels = []
    dataset = torchvision.datasets.CIFAR10(root="./", train=train, download=True,
                                           transform=torchvision.transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    for i, data in enumerate(tqdm(dataloader)):
        img = toPIL(data[0][0])
        img.save(os.path.join(output_dir, f'{i}.JPEG'))
        img.close()
        labels.append(data[1][0])
    np.savetxt(os.path.join(output_dir, f'labels.txt'), np.asarray(labels))


if __name__ == "__main__":
    save("./cifar10_data/val", False)
    #save("./cifar10_data/train", True)
