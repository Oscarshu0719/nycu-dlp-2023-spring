from datetime import datetime, timedelta
import numpy as np
import os
from pathlib import Path
import random
from timeit import default_timer as timer
from tqdm.auto import tqdm as tqdm
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataloader import BciDataLoader as DataLoader
from src.models.deep_conv_net import DeepConvNet
from src.models.eegnet import Eegnet
from src.utils import utils


class Solver(object): 
    def __init__(self, config: dict) -> None:
        if config.seed != None:
            self.fix_seed(config.seed)
            
        # Data loader.   
        data_loader = DataLoader(config.dir_dataset)
        [self.train_loader, self.test_loader] = data_loader.get_data_loader(
            batch_size=config.batch_size, num_workers=config.num_workers)
        
        # GPUs.
        self.device = torch.device(f'cuda:{config.gpu[0]}' if torch.cuda.is_available() else 'cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, config.gpu)))
        
        print(f'***\nCurrently using {self.device}...\n***\n')
        
        # Training configs.
        self.model = config.model
        self.num_epochs = config.num_epochs
        self.lr = config.lr
        self.optim = config.optim
        self.dropout = config.dropout
        self.batch_size = config.batch_size
        
        # Misc.
        self.timestamp = utils.get_timestamp(datetime.now())
        self.epoch_log = config.epoch_log
        self.is_plot_saving = config.is_plot_saving
        self.path_acc = config.path_acc
        
        if self.is_plot_saving: 
            self.mkdirs(config)
        
        # Build model.
        self.build_model(config)
        
        # Load model.
        if config.resume_ckpt.strip() != '':
            self.load_model(config.resume_ckpt)
        
    def train_all(self) -> None: 
        for setup in self.models:
            model = setup['model']
            optim = setup['optim']
            scheduler = setup['scheduler']
            criterion = setup['criterion']
            act = setup['act']
            
            self.train(model, optim, scheduler, criterion, act)
            
        # Plot.
        path = (f'{self.model}_{self.optim}'
                f'_{self.batch_size}_{self.lr}_{self.dropout}')
        model_loss = []
        model_loss += self.model_train_loss
        model_loss += self.model_test_loss
        utils.plot_result(
            model_loss, self.num_epochs, Path(self.dir_output, f'{path}_loss.png'), 
            'Loss', self.labels, 'upper right')
        
        model_acc = []
        model_acc += self.model_train_acc
        model_acc += self.model_test_acc
        utils.plot_result(
            model_acc, self.num_epochs, Path(self.dir_output, f'{path}_acc.png'), 
            'Accuracy', self.labels, 'lower right')
        
        id = (f'{self.model}_{self.optim}'
            f'_{self.batch_size}_{self.lr}_{self.dropout}')
        with open(self.path_acc, 'a') as file:
            for act, acc in zip(self.acts, self.model_train_acc):
                file.write(f'({id}_{act}) training accuracy: {max(acc) * 100:.2f}%.\n')
            for act, acc in zip(self.acts, self.model_test_acc):
                file.write(f'({id}_{act}) testing accuracy: {max(acc) * 100:.2f}%.\n')
                
            file.write(f'{"-" * 67}\n')
    
    def eval_all(self) -> None:
        num_test_data = len(self.test_loader.dataset)
        
        for setup in self.models:
            model = setup['model']
            criterion = setup['criterion']
            act = setup['act']
            
            model.eval()
            with torch.no_grad():
                test_total_loss = 0
                test_total_acc = 0
                for data, label in self.test_loader:
                    data = data.to(self.device)
                    label = label.to(self.device)
                    
                    pred = model(data)
                    
                    loss = criterion(pred, label)
                    test_total_loss += loss
                    test_total_acc += self.cal_acc(label, pred)
                
                test_mean_loss = test_total_loss / num_test_data
                test_mean_acc = test_total_acc / num_test_data
                
                print(f'{act:>9}: test_loss: {test_mean_loss:>7.5f}, test_acc: {test_mean_acc:>4.2f}.')
        
    def train(self, model, optim, scheduler, criterion, act: str) -> None:
        num_train_data = len(self.train_loader.dataset)
        num_test_data = len(self.test_loader.dataset)
        
        if self.is_plot_saving: 
            epoch_train_loss = []
            epoch_train_acc = []
            epoch_test_loss = []
            epoch_test_acc = []
            
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
                for data, label in self.test_loader:
                    data = data.to(self.device)
                    label = label.to(self.device)
                    
                    pred = model(data)
                    
                    loss = criterion(pred, label)
                    test_total_loss += loss
                    test_total_acc += self.cal_acc(label, pred)
                
                test_mean_loss = test_total_loss / num_test_data
                test_mean_acc = test_total_acc / num_test_data
                
            if self.is_plot_saving: 
                epoch_train_loss.append(train_mean_loss.item())
                epoch_train_acc.append(train_mean_acc)
                epoch_test_loss.append(test_mean_loss.item())
                epoch_test_acc.append(test_mean_acc)
                
            if (epoch + 1) % self.epoch_log == 0:
                print((f'Epoch [{epoch + 1:>3}], '
                       f'train_loss: {train_mean_loss:>7.5f}, train_acc: {train_mean_acc:>4.2f}, '
                       f'test_loss: {test_mean_loss:>7.5f}, test_acc: {test_mean_acc:>4.2f}.'))
        
        time_end = timer()
        
        self.model_train_loss.append(epoch_train_loss)
        self.model_train_acc.append(epoch_train_acc)
        self.model_test_loss.append(epoch_test_loss)
        self.model_test_acc.append(epoch_test_acc)

        print(f'\n***\nElapsed time (hh:mm:ss.ms): {timedelta(seconds=time_end - time_start)}.\n***')
        
        self.save_model(model, act, self.num_epochs)
        
    def build_model(self, config: dict) -> None: 
        self.acts = ['relu', 'leakyrelu', 'elu']
        
        # Model.
        model = Eegnet if config.model == 'eegnet' else DeepConvNet
        self.models = [
            {'model': model(act_name=act, dropout=self.dropout).to(self.device), 
             'act': act}
            for act in self.acts
        ]
        
        # Optimizer, scheduler, and loss function.
        for setup in self.models:
            model = setup['model']
            
            if config.optim == 'adam': 
                optim = torch.optim.Adam(
                    model.parameters(), lr=self.lr)
            elif config.optim == 'adamw':
                optim = torch.optim.AdamW(
                    model.parameters(), lr=self.lr)
            else: # 'sgd'.
                optim = torch.optim.SGD(
                    model.parameters(), lr=self.lr * 100, momentum=0.9)
            
            setup['optim'] = optim
            setup['scheduler'] = CosineAnnealingLR(optim, T_max=5, eta_min=0)
            setup['criterion'] = nn.CrossEntropyLoss()
            
        if config.is_model_printing:
            self.print_model()
            
        # Loss.
        self.model_train_loss = []
        self.model_train_acc = []
        self.model_test_loss = []
        self.model_test_acc = []
        
        # Plot.
        self.labels = [
            f'train_{act}'
            for act in self.acts
        ]
        self.labels += [
            f'test_{act}'
            for act in self.acts
        ]
        
    def load_model(self, dir_ckpt) -> None:
        try: 
            for act, setup in zip(self.acts, self.models): 
                setup['model'].load_state_dict(
                    torch.load(Path(dir_ckpt, f'{act}_{self.num_epochs}.pt'), map_location='cpu'))
                setup['model'] = setup['model'].to(self.device)
            
            print(f'Finished loading model from {dir_ckpt}...\n')
        except Exception as msg:
            print(f'Failed to load checkpoint from {dir_ckpt}. Message: \n{msg}.\n')
        
    def save_model(self, model, act: str, epoch: int) -> None:
        torch.save(model.state_dict(), 
            Path(self.dir_model, f'{act}_{epoch}.pt'))
        
    def cal_acc(self, real: torch.tensor, pred: torch.tensor) -> float: 
        return (torch.max(pred, 1)[1] == real).sum().item()
            
    def print_model(self) -> None: 
        model = self.models[0]['model']
        
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
        path = (f'{self.timestamp}_{config.model}_{self.optim}'
                f'_{config.batch_size}_{self.lr}_{self.dropout}')
        
        if config.mode == 'train':
            self.dir_output = Path(config.dir_output, path)
            if not self.dir_output.is_dir(): 
                self.dir_output.mkdir(parents=True, exist_ok=True)
            
            self.dir_model = Path(config.dir_model, path)
            if not self.dir_model.is_dir(): 
                self.dir_model.mkdir(parents=True, exist_ok=True)
            