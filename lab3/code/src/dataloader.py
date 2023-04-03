import numpy as np
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


class BciDataLoader(object):
    def __init__(self, dir_dataset: str) -> None:
        self.training_set = self.BciTrainingDataset(dir_dataset)
        self.testing_set = self.BciTestingDataset(dir_dataset)
        
    def get_data_loader(self, batch_size: int, num_workers: int, shuffle=True) -> list[DataLoader, DataLoader]:
        return [
            DataLoader(self.training_set, batch_size=batch_size, 
                shuffle=shuffle, num_workers=num_workers), 
            DataLoader(self.testing_set, batch_size=batch_size, 
                shuffle=shuffle, num_workers=num_workers)]
    
    class BciTrainingDataset(Dataset): 
        def __init__(self, dir_dataset: str) -> None:
            """
            train_data: (1080, 1, 2, 750).
            train_label: (1080,).
            """
            super().__init__()

            # Convert from double to float.
            S4b_train = np.load(Path(dir_dataset, 'S4b_train.npz'))
            X11b_train = np.load(Path(dir_dataset, 'X11b_train.npz'))
            
            train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
            train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
            
            train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
            train_label = train_label - 1
            
            mask = np.where(np.isnan(train_data))
            train_data[mask] = np.nanmean(train_data)
            
            self.dataset = train_data.astype(np.float32) # Float.
            self.label = train_label.astype(np.int64) # Long.
            self.num_samples = self.dataset.shape[0]
            
        def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
            return self.dataset[idx], self.label[idx]

        def __len__(self) -> int:
            return self.num_samples
        
    class BciTestingDataset(Dataset): 
        def __init__(self, dir_dataset: str) -> None:
            """
            test_data: (1080, 1, 2, 750).
            test_label: (1080,).
            """
            super().__init__()

            S4b_test = np.load(Path(dir_dataset, 'S4b_test.npz'))
            X11b_test = np.load(Path(dir_dataset, 'X11b_test.npz'))
            
            test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
            test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)
            
            test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))
            test_label = test_label - 1
            
            mask = np.where(np.isnan(test_data))
            test_data[mask] = np.nanmean(test_data)
            
            self.dataset = test_data.astype(np.float32) # Float.
            self.label = test_label.astype(np.int64) # Long.
            self.num_samples = self.dataset.shape[0]
            
        def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
            return self.dataset[idx], self.label[idx]

        def __len__(self) -> int:
            return self.num_samples
