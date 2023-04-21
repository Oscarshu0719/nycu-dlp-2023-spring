from datetime import datetime, timedelta
import numpy as np
import os
from pathlib import Path
import random
from timeit import default_timer as timer
from tqdm.auto import tqdm as tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from src.dataloader import RetinopathyDataLoader as DataLoader
from src.resnet import ResNet_18, ResNet_50
from src.utils import utils

"""
Eval: load pretrained model weight only.
"""
class Solver(object): 
    def __init__(self, config: dict) -> None:
        if config.seed != None:
            self.fix_seed(config.seed)
            
        # Data loader.   
        data_loader = DataLoader(config.dir_dataset)
        [self.train_loader, self.test_loader] = data_loader.get_data_loader(
            batch_size=config.batch_size, num_workers=config.num_workers)
        self.test_labels = data_loader.get_test_labels()
        
        # GPUs.
        self.device = torch.device(f'cuda:{config.gpu[0]}' if torch.cuda.is_available() else 'cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, config.gpu)))
        
        print(f'***\nCurrently using {self.device}...\n***\n')
        
        # Training configs.
        self.model = config.model
        self.num_epochs = config.num_epochs
        self.lr = config.lr
        self.optim = config.optim
        self.weight_decay = config.weight_decay
        self.batch_size = config.batch_size
        
        # Misc.
        self.timestamp = utils.get_timestamp(datetime.now())
        self.epoch_log = config.epoch_log
        self.is_plot_saving = config.is_plot_saving
        
        if self.is_plot_saving: 
            self.mkdirs(config)
        
        # Build model.
        self.build_model(config)
        
    def train_all(self) -> None: 
        for setup in self.setups:
            model = setup['model']
            optim = setup['optim']
            scheduler = setup['scheduler']
            criterion = setup['criterion']
            tag = setup['tag']
            
            self.train(model, optim, scheduler, criterion, tag)
            
        # Plot.
        path = (f'{self.model}_{self.optim}'
                f'_{self.batch_size}_{self.lr}_{self.weight_decay}')
        
        model_loss = []
        model_loss += self.model_train_loss
        model_loss += self.model_test_loss
        utils.plot_result(
            model_loss, self.num_epochs, Path(self.dir_output, f'{path}_loss.png'), 
            'Loss', self.plot_labels, 'upper right')
        
        model_acc = []
        model_acc += self.model_train_acc
        model_acc += self.model_test_acc
        utils.plot_result(
            model_acc, self.num_epochs, Path(self.dir_output, f'{path}_acc.png'), 
            'Accuracy', self.plot_labels, 'upper left')
        
    def eval_all(self) -> None:
        num_test_data = len(self.test_loader.dataset)
        
        # Only evaluate pretrained model.
        model = self.setups[0]['model']
        criterion = self.setups[0]['criterion']
            
        model.eval()
        with torch.no_grad():
            test_total_loss = 0
            test_total_acc = 0
            for data, label in tqdm(self.test_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                
                pred = model(data)
                
                loss = criterion(pred, label)
                test_total_loss += loss
                test_total_acc += self.cal_acc(label, pred)
            
            test_mean_loss = test_total_loss / num_test_data
            test_mean_acc = test_total_acc / num_test_data
            
            print(f'test_loss: {test_mean_loss:>7.5f}, test_acc: {test_mean_acc:>4.2f}.')
        
    def train(self, 
            model: nn.Module, optim: torch.optim, 
            scheduler: torch.optim.lr_scheduler, criterion: nn.modules.loss, tag: str) -> None:
        num_train_data = len(self.train_loader.dataset)
        num_test_data = len(self.test_loader.dataset)
        
        if self.is_plot_saving: 
            epoch_train_loss = []
            epoch_train_acc = []
            epoch_test_loss = []
            epoch_test_acc = []
            
        # Store labels of best iteration.
        best_acc = 0
        best_pred_labels = None
        
        time_start = timer()
            
        for epoch in tqdm(range(self.num_epochs)): 
            # Training.
            model.train()
            
            train_total_loss = 0
            train_total_acc = 0
            for data, label in self.train_loader:
                data = data.to(self.device)
                label = label.to(self.device)
                
                pred = model(data)
                
                optim.zero_grad()
                loss = criterion(pred, label)
                
                loss.backward()
                optim.step()
                
                train_total_loss += loss
                train_total_acc += self.cal_acc(label, pred)
                
            scheduler.step()
            
            train_mean_loss = train_total_loss / num_train_data
            train_mean_acc = train_total_acc / num_train_data
            
            # Testing.
            model.eval()
            with torch.no_grad():
                test_total_loss = 0
                test_total_acc = 0
                pred_labels = np.array([], dtype=np.int32)
                for data, label in self.test_loader:
                    data = data.to(self.device)
                    label = label.to(self.device)
                    
                    pred = model(data)
                    
                    loss = criterion(pred, label)
                    test_total_loss += loss
                    test_total_acc += self.cal_acc(label, pred)
                    
                    pred_labels = np.concatenate((pred_labels, torch.max(pred, 1)[1].to('cpu').numpy()))
                
                test_mean_loss = test_total_loss / num_test_data
                test_mean_acc = test_total_acc / num_test_data
                
                if test_mean_acc > best_acc:
                    best_acc = test_mean_acc
                    best_pred_labels = pred_labels
                
            if self.is_plot_saving: 
                epoch_train_loss.append(train_mean_loss.item())
                epoch_train_acc.append(train_mean_acc)
                epoch_test_loss.append(test_mean_loss.item())
                epoch_test_acc.append(test_mean_acc)
                
            if (epoch + 1) % self.epoch_log == 0:
                print((f'Epoch [{epoch + 1:>3}], '
                       f'train_loss: {train_mean_loss:>7.5f}, train_acc: {train_mean_acc:>4.2f}, '
                       f'test_loss: {test_mean_loss:>7.5f}, test_acc: {test_mean_acc:>4.2f}.'))
                
        utils.confusion_matrix(
            self.test_labels, best_pred_labels, Path(self.dir_output, f'confusion_matrix_{tag}.png'), self.model)
                
        time_end = timer()
        
        self.model_train_loss.append(epoch_train_loss)
        self.model_train_acc.append(epoch_train_acc)
        self.model_test_loss.append(epoch_test_loss)
        self.model_test_acc.append(epoch_test_acc)

        print(f'\n***\nElapsed time (hh:mm:ss.ms): {timedelta(seconds=time_end - time_start)}.\n***')
        
        self.save_model(model, self.num_epochs, tag)
        
    def build_model(self, config: dict) -> None: 
        if config.mode == 'train': 
            # Model.
            if config.model == 'resnet18': 
                self.setups = [{
                    'model': ResNet_18().to(self.device), 
                    'tag': 'trained'
                }]
                pretrained = models.resnet18
            else: # 'resnet50'.
                self.setups = [{
                    'model': ResNet_50().to(self.device), 
                    'tag': 'trained'
                }]
                pretrained = models.resnet50
            
            self.setups.append({
                'model': pretrained(weights='DEFAULT').to(self.device), 
                'tag': 'pretrained'
            })
            
            # Optimizer, scheduler, and loss function.
            for setup in self.setups:
                model = setup['model']
                
                if config.optim == 'sgd': 
                    optim = torch.optim.SGD(
                        model.parameters(), 
                        lr=self.lr, momentum=0.9, weight_decay=config.weight_decay)
                elif config.optim == 'adamw':
                    optim = torch.optim.AdamW(
                        model.parameters(), 
                        lr=self.lr, weight_decay=config.weight_decay)
                else: # 'adam'.
                    optim = torch.optim.Adam(
                        model.parameters(), 
                        lr=self.lr, weight_decay=config.weight_decay)
                
                setup['optim'] = optim
                setup['scheduler'] = CosineAnnealingLR(optim, T_max=5, eta_min=0)
                setup['criterion'] = nn.CrossEntropyLoss()
        else: # 'eval'
            # ONLY build pretrained model.
            self.setups = [{
                'model': self.load_model(
                    models.resnet18() if config.model == 'resnet18' else models.resnet50(), 
                    Path(config.dir_model, config.resume_ckpt)), 
                'criterion': nn.CrossEntropyLoss()
            }]
            
        if config.is_model_printing:
            self.print_model()
            
        # Loss.
        self.model_train_loss = []
        self.model_train_acc = []
        self.model_test_loss = []
        self.model_test_acc = []
        
        # Plot.
        self.plot_labels = [
            'train_mine', 'train_pretrained', 
            'test_mine', 'test_pretrained'
        ]
        
    def load_model(self, model: nn.Module, path: str) -> None:
        try: 
            model.load_state_dict(
                torch.load(path, map_location='cpu'))
            model = model.to(self.device)
            
            print(f'Finished loading model from {path}...\n')
            
            return model
            
        except Exception as msg:
            print(f'Failed to load checkpoint from {path}. Message: \n{msg}.\n')
        
    def save_model(self, model: nn.Module, epoch: int, tag: str) -> None:
        torch.save(model.state_dict(), 
            Path(self.dir_model, f'{tag}_{epoch}.pt'))
        
    def cal_acc(self, real: torch.tensor, pred: torch.tensor) -> float: 
        return (torch.max(pred, 1)[1] == real).sum().item()
            
    def print_model(self) -> None: 
        for setup in self.setups: 
            model = setup['model']
            
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            
            print(f'\n***\nNumber of parameters: {num_params}.\n***')
            print(model)
            print()
        
    def fix_seed(self, seed) -> None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        
    def mkdirs(self, config: dict) -> None: 
        path = (f'{self.timestamp}_{config.model}_{config.optim}'
                f'_{config.batch_size}_{config.lr}_{config.weight_decay}')
        
        if config.mode == 'train':
            self.dir_output = Path(config.dir_output, path)
            if not self.dir_output.is_dir(): 
                self.dir_output.mkdir(parents=True, exist_ok=True)
            
            self.dir_model = Path(config.dir_model, path)
            if not self.dir_model.is_dir(): 
                self.dir_model.mkdir(parents=True, exist_ok=True)
            