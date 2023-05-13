import csv
import numpy as np
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional


def get_data_loaders(
        dir_dataset: str, num_cond: int, num_pred: int, num_eval: int, 
        batch_size=12, shuffle=True, num_workers=8, transform: Optional[list]=[
            transforms.ToTensor()]) -> list[DataLoader]:
    
    transform = transform if transform else []
    transform = transforms.Compose(transform)

    # 43008.
    train_dataset = __BairRobotPushingDataset(
        dir_dataset, num_cond, num_pred, num_eval, 
        mode='train', transform=transform)
    # 256.
    valid_dataset = __BairRobotPushingDataset(
        dir_dataset, num_cond, num_pred, num_eval, 
        mode='validate', transform=transform)
    # 256.
    test_dataset = __BairRobotPushingDataset(
        dir_dataset, num_cond, num_pred, num_eval, 
        mode='test', transform=transform)
    
    return [
        DataLoader(train_dataset, batch_size=batch_size, 
            shuffle=shuffle, num_workers=num_workers, drop_last=True, pin_memory=True), 
        DataLoader(valid_dataset, batch_size=batch_size, 
            shuffle=shuffle, num_workers=num_workers, drop_last=True, pin_memory=True), 
        DataLoader(test_dataset, batch_size=batch_size, 
            shuffle=shuffle, num_workers=num_workers, drop_last=True, pin_memory=True)]

class __BairRobotPushingDataset(Dataset):
    def __init__(self, 
            dir_dataset: str, num_cond: int, num_pred: int, num_eval: int, 
            transform: transforms.Compose, mode='train'):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        
        self.dir_dataset = Path(dir_dataset, mode)
        self.len_seq = max(num_cond + num_pred, num_eval)
        self.transform = transform
        self.mode = mode
        self.ordered = mode != 'train'
        
        self.dirs = []
        for dir1 in os.listdir(self.dir_dataset):
            for dir2 in os.listdir(Path(self.dir_dataset, dir1)):
                self.dirs.append(Path(self.dir_dataset, dir1, dir2))
                
        self.idx = 0
        self.cur_dir = self.dirs[0]
        self.num_samples = len(self.dirs)
                
    def get_seq(self):
        if self.ordered:
            self.cur_dir = self.dirs[self.idx]
            self.idx = (self.idx + 1) % self.num_samples
        else:
            self.cur_dir = self.dirs[
                np.random.randint(self.num_samples)]
            
        img_seq = []
        for i in range(self.len_seq):
            fname = f'{self.cur_dir}/{i}.png'
            img = Image.open(fname)
            img_seq.append(self.transform(img))
        img_seq = torch.stack(img_seq)

        return img_seq
    
    def get_csv(self):
        with open(f'{self.cur_dir}/actions.csv', newline='') as csvfile:
            rows = csv.reader(csvfile)
            actions = []
            for i, row in enumerate(rows):
                if i == self.len_seq:
                    break
                action = [float(value) for value in row]
                actions.append(torch.tensor(action))
            
            actions = torch.stack(actions)
            
        with open(f'{self.cur_dir}/endeffector_positions.csv', newline='') as csvfile:
            rows = csv.reader(csvfile)
            positions = []
            for i, row in enumerate(rows):
                if i == self.len_seq:
                    break
                position = [float(value) for value in row]
                positions.append(torch.tensor(position))
            positions = torch.stack(positions)

        condition = torch.cat((actions, positions), axis=1)

        return condition
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        seq = self.get_seq()
        cond = self.get_csv()
        
        # [batch_size, len_seq, 3, 64, 64], [batch_size, len_seq, 7].
        return seq, cond
    
__all__ = ['get_data_loaders']
