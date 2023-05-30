import cv2
from datetime import datetime
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import random
from tensorboardX import SummaryWriter
import torch
from torchvision.utils import save_image, make_grid
from tqdm.auto import tqdm

from src.data_loader import get_test_labels, get_train_loader
from src.ddpm import DDPM
from src.resnet import ResNet


class Solver(object):
    def __init__(self, config: dict) -> None:
        self.timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        self.device = torch.device(f'cuda:{config.gpu[0]}' if torch.cuda.is_available() else 'cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, config.gpu)))
        
        print(f'***\nCurrently using {self.device}...\n***\n')
        
        if config.seed != None:
            self.fix_seed(config.seed)
            
        self.date_loader, self.num_classes, self.data_shape = get_train_loader(
            config.batch_size, config.dir_dataset, config.num_workers, True)
        
        is_training = config.mode == 'train'
        if not is_training: # 'test'.
            self.test_conds = get_test_labels(config.dir_dataset)
        
        self.build_model(config)
        self.mkdirs(config)
        
        self.epoch_start = 0
        ckpt = config.resume_ckpt.strip()
        if ckpt != '':
            self.load_model(ckpt)
            
        self.train_num_epochs = config.train_num_epochs
        self.lr = config.lr
        self.epoch_model_saving = config.epoch_model_saving
        self.epoch_img_saving = config.epoch_img_saving
        self.ws_test = [0.0, 0.5, 2.0] # strength of generative guidance
            
        if is_training: 
            self.train()
        else: # 'test'.
            self.eval()
            
    def train(self) -> None:
        iter = 0
        for epoch in range(self.train_num_epochs):
            print(f'Epoch {epoch + 1} ...')
             
            self.model.train()
            
            # Linear learning rate decay.
            new_lr = self.lr * (1 - epoch / self.train_num_epochs)
            self.optim.param_groups[0]['lr'] = new_lr
            
            self.writer.add_scalar('lr', new_lr, epoch)
            
            pbar = tqdm(self.date_loader)
            loss_ema = None
            for (img, cond) in pbar:
                self.optim.zero_grad()
                
                img = img.to(self.device)
                cond = cond.to(self.device)
                loss = self.model(img, cond)
                
                loss.backward()
                
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                pbar.set_description(f'loss: {loss_ema:.4f}')
                
                self.writer.add_scalar('train_loss_ema', loss_ema, iter)
                iter += 1
                                     
                self.optim.step()
                
            if (epoch + 1) % self.epoch_img_saving == 0 or epoch == int(self.train_num_epochs - 1):
                self.model.eval()
                with torch.no_grad():
                    n_sample = 2 * self.num_classes
                    for w in self.ws_test:
                        x_gen, x_gen_store = self.model.sample(
                            n_sample, self.data_shape, self.device, guide_w=w)
                        
                        self.plot_images(x_gen, epoch, n_sample, n_row=8, w=w)
                        self.plot_animation(x_gen_store, epoch, n_sample, w=w)
                            
                print()        
                    
            if (epoch + 1) % self.epoch_model_saving == 0:
                self.save_model(epoch + 1)
                
    def eval(self) -> None:
        self.model.eval()
        with torch.no_grad():
            for w in self.ws_test:
                x_gen = self.model.generate(
                    self.test_conds, self.test_conds.shape[0], 
                    self.data_shape, self.device, guide_w=w)
                
                score = self.model_eval.eval(x_gen, self.test_conds)
                print(f'Score with w={w}: {score:.2f}')
                
                self.plot_images(x_gen, self.epoch_start, x_gen.shape[0], 
                    n_row=4, w=w, mode='test')
                
    def plot_images(self, 
            gen_imgs, epoch, n_sample, n_row=8, w=0.0, mode='train') -> None:
        # Image.
        grid = make_grid(gen_imgs * -1 + 1, nrow=n_sample // n_row)
        
        if mode == 'train': 
            path = Path(self.dir_output, f'epoch{epoch + 1}_w{w}.png')
        else: # 'test'.
            path = Path(self.dir_output, f'test_w{w}.png')
            
        save_image(grid, path)
        
    def plot_animation(self, gen_imgs_store, epoch, n_sample, w=0.0) -> None:
        # Gif.
        fig, axs = plt.subplots(
            nrows=n_sample // self.num_classes, ncols=self.num_classes, 
            sharex=True, sharey=True, figsize=(8, 3))
        def animate_diff(i, x_gen_store):
            print(f'Generating animation frame {i + 1}/{x_gen_store.shape[0]} with guidance {w} ...', end='\r')
            plots = []
            for row in range(n_sample // self.num_classes):
                for col in range(self.num_classes):
                    axs[row, col].clear()
                    axs[row, col].set_xticks([])
                    axs[row, col].set_yticks([])
                    plots.append(axs[row, col].imshow(
                        -x_gen_store[i, (row * self.num_classes) + col, 0], 
                        cmap='gray', vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
            return plots
        ani = FuncAnimation(
            fig, animate_diff, fargs=[gen_imgs_store], 
            interval=200, blit=False, repeat=True, frames=gen_imgs_store.shape[0])    
        ani.save(Path(self.dir_output, f'epoch{epoch + 1}_w{w}.gif'), dpi=100, writer=PillowWriter(fps=5))
    
    def build_model(self, config: dict) -> None:
        self.model = DDPM(self.num_classes, self.device).to(self.device)
        
        if config.mode == 'test': 
            self.model_eval = ResNet(
                self.num_classes, config.dir_dataset, self.device)
        
        if config.optim == 'adam': 
            self.optim = torch.optim.Adam(
                self.model.parameters(), lr=config.lr)
            
    def load_model(self, path: str) -> None: 
        path = Path(path)
        try: 
            self.epoch_start = int(path.stem.split('_')[-1].split('.')[0])
            
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
            self.model = self.model.to(self.device)
            
            print(f'Finished loading model from {path} (epoch {self.epoch_start + 1}) ...')
        except Exception as msg:
            print(f'Failed to load checkpoint from {path}. Message: \n{msg}')
        
    def save_model(self, epoch: int) -> None:
        torch.save(self.model.state_dict(), 
            Path(self.dir_model, f'{self.timestamp}_{epoch}.pt'))
    
    def mkdirs(self, config: dict) -> None: 
        self.path = f'{self.timestamp}_{config.mode}'
        
        if config.mode == 'train':
            # Directory will be created by the function if not exists.
            self.writer = SummaryWriter(
                log_dir=Path(config.dir_writer, self.path))
            
            self.dir_model = Path(config.dir_model, self.path)
            
            dir_model = Path(self.dir_model)
            if not dir_model.is_dir(): 
                dir_model.mkdir(parents=True, exist_ok=True)
                
        self.dir_output = Path(config.dir_output, self.path)

        dir_output = Path(self.dir_output)
        if not dir_output.is_dir(): 
            dir_output.mkdir(parents=True, exist_ok=True)
            
    def fix_seed(self, seed) -> None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        