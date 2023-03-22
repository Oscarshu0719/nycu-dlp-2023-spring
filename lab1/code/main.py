import argparse
from datetime import datetime
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter

from src.data_loader import data_loader
from src.loss import loss
from src.model import Model
from src.utils import utils

"""
TODO: Hidden layers back-propgation.
"""

class Solver(object):
    def __init__(self, config: dict) -> None:
        self.load_data(config.dataset, config.num_data)
        
        self.build_model(
            config.dim_hidden, config.num_hidden, 
            config.lr, config.activation, config.optim, config.seed)
        
        """
        Training configs.
        """
        self.num_epochs = config.num_epochs
        self.acc_goal = config.acc_goal
        
        """
        Logging configs.
        """
        self.epoch_checking = config.epoch_checking
        self.epoch_logging = config.epoch_logging
        self.is_writer_logging = config.is_writer_logging
        self.is_image_saving = config.is_image_saving
        self.dir_output = config.dir_output
        
        # Create directory if it doesn't exist.
        dir_output = Path(config.dir_output)
        if not dir_output.is_dir(): 
            dir_output.mkdir(parents=True, exist_ok=True)
            
        self.timestamp = utils.get_timestamp(datetime.now())
        if config.is_writer_logging: 
            self.writer = SummaryWriter(
                log_dir=Path(config.dir_writer, self.timestamp))
        
        self.train()
        
    def load_data(self, dataset: str, num_data: int) -> None:
        if dataset == 'linear': 
            self.real_x, self.real_y = data_loader.generate_linear(n=num_data) # [n, 2], [n, 1].
        else: # 'xor'.
            self.real_x, self.real_y = data_loader.generate_XOR_easy() # [21, 2], [21, 1].

    def build_model(self, 
            dim_hidden: int, num_hidden: int, 
            lr: float, activation: str, optimizer: str, seed: int) -> None: 
        layers = [dim_hidden] * num_hidden
        layers.insert(0, self.real_x.shape[1])
        layers.append(1)
        
        self.model = Model(layers, 
            lr=lr, activation=activation, optimizer=optimizer, seed=seed)
        
    def train(self) -> None:
        for epoch in tqdm(range(self.num_epochs)):
            pred_y = self.model.forward(self.real_x)
            loss_epoch = loss.cross_entropy(self.real_y, pred_y)
            acc, pred_cls = self.eval(pred_y)
            self.model.backward(self.real_x, self.real_y)
            
            if self.is_writer_logging: 
                self.writer.add_scalar('loss', loss_epoch.item(), epoch)
                self.writer.add_scalar('accuracy', acc.item(), epoch)
            
            if (epoch + 1) % self.epoch_logging == 0:
                print(f'Epoch: {epoch + 1}, loss: {loss_epoch.item():.6f}, acc: {acc:.6f}.')
            
            if (epoch + 1) % self.epoch_checking == 0: 
                if acc == self.acc_goal:
                    print(f'\n[INFO]: Accuracy reaches {acc}. Early stop at epoch {epoch + 1}. Final prediction: ')
                    print(pred_y.squeeze(1).T)
                    print()
                    
                    if self.is_image_saving:
                        utils.show_result(self.real_x, self.real_y, pred_cls, 
                            is_saving=self.is_image_saving, path=Path(self.dir_output, f'{self.timestamp}.png'))
                    
                    return
        
        print(f'\n[WARN]: Accuracy only reaches {acc:.6f} at epoch {self.num_epochs}. Current prediction: ')
        print(pred_y.squeeze(1).T)
        print()
        
        if self.is_image_saving: 
            utils.show_result(self.real_x, self.real_y, pred_cls, 
                is_saving=self.is_image_saving, path=Path(self.dir_output, f'{self.timestamp}.png'))
                
    def eval(self, pred_y: np.ndarray) -> None: 
        pred_cls = np.array(pred_y, copy=True)
        
        pred_cls[pred_cls > 0.5] = 1
        pred_cls[pred_cls <= 0.5] = 0
        
        acc = np.sum((pred_cls == self.real_y)) / self.real_y.shape[0]
        
        return acc, pred_cls

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    """
    Data configs.
    """
    parser.add_argument('--dataset', type=str, default='linear', choices=['linear', 'xor'],  
        help='Dataset. Options: "linear" (default) or "xor".')
    parser.add_argument('--num_data', type=int, default=100, 
        help='Data size.')
    
    """
    Dir configs.
    """
    parser.add_argument('--dir_writer', type=str, default='./runs/', 
        help='Directory for saving SummaryWriter outputs.')
    parser.add_argument('--dir_output', type=str, default='./outputs/', 
        help='Directory for saving the image for comparison.')
    
    """
    Logging configs.
    """
    parser.add_argument('--epoch_checking', type=int, default=100, 
        help='Number of epochs for checking if accuracy reaches the goal.')
    parser.add_argument('--epoch_logging', type=int, default=2000, 
        help='Number of epochs for logging.')
    parser.add_argument('--is_writer_logging', type=lambda x: x == 'True', default=False, 
        help='Determine if SummaryWriter is logging.')
    parser.add_argument('--is_image_saving', type=lambda x: x == 'True', default=False, 
        help='Determine if the image for comparison is saving.')
    
    """
    Model configs.
    """
    parser.add_argument('--num_hidden', type=int, default=2, 
        help='Number of hidden layers.')
    parser.add_argument('--dim_hidden', type=int, default=4, 
        help='Dimension of hidden layers.')
    
    """
    Training configs.
    """
    parser.add_argument('--num_epochs', type=int, default=30000, 
        help='Number of epochs to train.')
    parser.add_argument('--seed', type=int, default=42, 
        help='Random seed.')
    parser.add_argument('--lr', type=float, default=1e-2, 
        help='Learning rate.')
    parser.add_argument('--optim', type=str, default='vanilla', choices=['vanilla', 'adam'],  
        help='Optimizer. Options: "vanilla" (default) or "adam".')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'leaky_relu'],  
        help='Activation function. Options: "relu" (default), or "leaky_relu".')
    parser.add_argument('--acc_goal', type=float, default=1., 
        help='Early stop when reaching accuracy goal.')
    
    Solver(parser.parse_args())
