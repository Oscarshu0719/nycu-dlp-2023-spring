import pandas as pd
import PIL

import numpy as np
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from typing import Optional

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.utils import utils

"""
How to download images?

1. Download directly: 
    - URL: https://drive.google.com/drive/folders/1Kh2Kl9-BqP4kEk9n59mxOw0Z1pJy0wuv

2. Download with `wget`: 
    - new_train.zip: 
        - URL: https://drive.google.com/file/d/1uNPik6rtEWpyYsOOWUHffMjrWnXOfcPC/view?usp=share_link 
        - ID: 1uNPik6rtEWpyYsOOWUHffMjrWnXOfcPC
    - new_test.zip: 
        - URL: https://drive.google.com/file/d/1D4iegBuFmtLwbzqbbcChWVBcN5PRb41S/view?usp=share_link
        - ID: 1D4iegBuFmtLwbzqbbcChWVBcN5PRb41S
    - Command: 
    ```bash
    # Download .zip file by its ID and unzip it to directory `DIR_UNZIP`.
    # Use new_train.zip as example.
    FILEID="1uNPik6rtEWpyYsOOWUHffMjrWnXOfcPC"
    FILENAME="train.zip"
    FILE_TMP="/tmp/cookies.txt"
    DIR_UNZIP="./lab4/datasets/"

    wget --load-cookies $FILE_TMP \
        "https://docs.google.com/uc?export=download&confirm= \
        $(wget --quiet --save-cookies $FILE_TMP --keep-session-cookies --no-check-certificate \
        "https://docs.google.com/uc?export=download&id=${FILEID}" -O- | \
        sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $FILENAME && \ 
    unzip $FILENAME -d $DIR_UNZIP && \
    rm $FILE_TMP $FILENAME
    ``
"""
    

class RetinopathyDataLoader(object):
    def __init__(self, dir_dataset: str, transform: Optional[list]=[
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5), 
            transforms.Resize((512, 512)), 
            transforms.ToTensor()
        ]) -> None:
        
        self.transform = transform if transform else []
        self.transform = transforms.Compose(self.transform)
        
        self.training_set = self.RetinopathyTrainingDataset(
            dir_dataset, self.transform)
        self.testing_set = self.RetinopathyTestingDataset(
            dir_dataset, transforms.Compose([
               transforms.Resize((512, 512)), transforms.ToTensor()]))
        
    def get_data_loader(self, 
            batch_size: int, num_workers: int, shuffle=True) -> list[DataLoader, DataLoader]:
        return [
            DataLoader(self.training_set, batch_size=batch_size, 
                shuffle=shuffle, num_workers=num_workers), 
            DataLoader(self.testing_set, batch_size=batch_size, 
                shuffle=shuffle, num_workers=num_workers)]
        
    def get_test_labels(self) -> np.ndarray:
        return self.testing_set.label
    
    class RetinopathyTrainingDataset(Dataset): 
        def __init__(self, 
                dir_dataset: str, transform: transforms.Compose) -> None:
            """
            train_img: (28100,).
            train_lbl: (28100,).
            """
            super().__init__()
            self.dir_dataset = dir_dataset
            self.transform = transform

            train_img = pd.read_csv(Path(dir_dataset, 'train_img.csv'), header=None)
            train_lbl = pd.read_csv(Path(dir_dataset, 'train_label.csv'), header=None)
            
            self.dataset = np.squeeze(train_img.values)
            self.label = np.squeeze(train_lbl.values).astype(np.int64)
            
            self.num_samples = self.dataset.shape[0]
            
        def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
            img = utils.center_crop(
                PIL.Image.open(Path(self.dir_dataset, 'train', f'{self.dataset[idx]}.jpeg')))
            img = self.transform(img)
            lbl = self.label[idx]
            
            return img, lbl

        def __len__(self) -> int:
            return self.num_samples
        
    class RetinopathyTestingDataset(Dataset): 
        def __init__(self, 
                dir_dataset: str, transform: transforms.Compose) -> None:
            """
            test_img: (7026,).
            test_lbl: (7026,).
            """
            super().__init__()
            self.dir_dataset = dir_dataset
            self.transform = transform

            test_img = pd.read_csv(Path(dir_dataset, 'test_img.csv'), header=None)
            test_lbl = pd.read_csv(Path(dir_dataset, 'test_label.csv'), header=None)
            
            self.dataset = np.squeeze(test_img.values)
            self.label = np.squeeze(test_lbl.values).astype(np.int64)
            
            self.num_samples = self.dataset.shape[0]

        def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
            img = utils.center_crop(
                PIL.Image.open(Path(self.dir_dataset, 'test', f'{self.dataset[idx]}.jpeg')))
            img = self.transform(img)
            lbl = self.label[idx]
            
            return img, lbl

        def __len__(self) -> int:
            return self.num_samples
