from datetime import datetime
import imageio
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from scipy import signal
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms


class __Utils(object):
    _instance = None 
    def __new__(cls): 
        if cls._instance is None: 
            cls._instance = super().__new__(cls) 
        return cls._instance
    
    def __int__(self) -> None:
        pass
    
    def get_timestamp(self, timestamp=datetime.now()) -> str:
        return timestamp.strftime('%y%m%d_%H%M%S')
    
    def init_weights(self, m: nn.Module) -> None:
        classname = m.__class__.__name__
        
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
            
    def get_kld(self, 
            mu: torch.FloatTensor, logvar: torch.FloatTensor, batch_size: int) -> float: 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= batch_size
        
        return KLD
            
    def get_mse(self, 
            x: torch.FloatTensor, y: torch.FloatTensor) -> torch.FloatTensor:
        loss = (x - y) ** 2
        loss = loss.mean()
        
        return loss
            
    def __finn_psnr(self, x, y, data_range=1.):
        mse = ((x - y)**2).mean()
        return 20 * math.log10(data_range) - 10 * math.log10(mse)
    
    def __fspecial_gauss(self, size, sigma):
        x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        return g / g.sum()
            
    def __finn_ssim(self, img1, img2, data_range=1., cs_map=False):
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        size = 11
        sigma = 1.5
        window = self.__fspecial_gauss(size, sigma)

        K1 = 0.01
        K2 = 0.03

        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2
        sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
        sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
        sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

        if cs_map:
            return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
        else:
            return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                (sigma1_sq + sigma2_sq + C2))
            
    def finn_eval_seq(self, gt, pred):
        T = len(gt)
        bs = gt[0].shape[0]
        ssim = np.zeros((bs, T))
        psnr = np.zeros((bs, T))
        mse = np.zeros((bs, T))
        for i in range(bs):
            for t in range(T):
                origin = gt[t][i].detach().cpu().numpy()
                predict = pred[t][i].detach().cpu().numpy()
                for c in range(origin.shape[0]):
                    res = self.__finn_ssim(origin[c], predict[c]).mean()
                    if math.isnan(res):
                        ssim[i, t] += -1
                    else:
                        ssim[i, t] += res
                    psnr[i, t] += self.__finn_psnr(origin[c], predict[c])
                ssim[i, t] /= origin.shape[0]
                psnr[i, t] /= origin.shape[0]
                mse[i, t] = self.get_mse(origin, predict)
                
        return psnr
    
    def plot_pred(self, 
            seq: torch.FloatTensor, seq_pred: torch.FloatTensor, epoch: int, 
            dir: str) -> None:
        """
        seq, seq_pred: 
            - [b, len_seq, 3, 64, 64], e.g., [128, 12, 3, 64, 64].
        """
        
        sample = 0
        plots = [[], []]
        for frame_idx in range(seq_pred.shape[1]):
            plots[0].append(seq[:, frame_idx][sample])
            plots[1].append(seq_pred[:, frame_idx][sample])
            
        self.plot_img(Path(dir, f'{epoch}.png'), plots)
        self.plot_animation(Path(dir, f'{epoch}.gif'), plots)
        
    def plot_img(self, path: str, seqs: list) -> None:
        transform = transforms.ToPILImage()
        
        num_samples = len(seqs)
        time = len(seqs[0])
        bg = Image.new('RGB', (time * 64, num_samples * 64), '#000000')

        for sample in range(num_samples):
            for t in range(time):
                img = transform(seqs[sample][t])
                x = t % time
                y = sample % num_samples
                
                bg.paste(img, (x * 64, y * 64))

        bg.save(path)
        
    def plot_animation(self, path: str, seqs: list) -> None:
        transform = transforms.ToPILImage()
        
        num_samples = len(seqs)
        time = len(seqs[0])
        gifs = []

        for t in range(time):
            bg = Image.new('RGB', (num_samples * 64, 64), '#000000')
            for sample in range(num_samples):
                img = transform(seqs[sample][t])
                x = sample % 30
                
                bg.paste(img, (x * 64, 0))

            gifs.append(bg)

        bg.save(path, save_all=True, append_images=gifs,
                optimize=False, duration=100, loop=0)
    
    def plot_result(self, 
            loss_list: list, psnr_list: list, kl_weights: list, tf_ratios: list, 
            epoch_start: int, num_epochs: int, epoch_check_psnr: int, dir: str) -> None:
        IMG_SIZE = (15, 15)
        FONT_SIZE = 18
        
        fig, ax1 = plt.subplots(figsize=IMG_SIZE)
        ax2 = ax1.twinx()
        ax1.set_xlabel('Epoch', fontsize=FONT_SIZE)
        ax1.set_ylabel('Score/Weight', fontsize=FONT_SIZE)
        ax2.set_ylabel('Loss', fontsize=FONT_SIZE)

        ax1.plot(range(epoch_start, epoch_start + num_epochs),
                tf_ratios, color='b', linestyle='--', label='tf_ratio')
        ax1.plot(range(epoch_start, epoch_start + num_epochs),
                kl_weights, color='g', linestyle='--', label='kl_weight')
        ax2.plot(range(epoch_start, epoch_start + num_epochs),
                loss_list, color='y', label='loss')
        fig.legend(fontsize=FONT_SIZE)
        fig.savefig(Path(dir, 'loss.png'))
        
        fig, ax1 = plt.subplots(figsize=IMG_SIZE)
        ax2 = ax1.twinx()
        ax1.set_xlabel('Epoch', fontsize=FONT_SIZE)
        ax1.set_ylabel('Score/Weight', fontsize=FONT_SIZE)
        ax2.set_ylabel('Loss', fontsize=FONT_SIZE)

        ax1.plot(range(epoch_start, epoch_start + num_epochs),
                tf_ratios, color='b', linestyle='--', label='tf_ratio')
        ax1.plot(range(epoch_start, epoch_start + num_epochs),
                kl_weights, color='g', linestyle='--', label='kl_weight')
        ax2.plot(range(epoch_start, epoch_start + num_epochs, epoch_check_psnr),
                psnr_list, color='r', linestyle='dotted', label='PSNR')
        fig.legend(fontsize=FONT_SIZE)
        fig.savefig(Path(dir, 'psnr.png'))
        
        print(f'[INFO]: Finish saving loss.png and psnr.png to {dir}...')
        
    def ssim_and_plot(self, 
            seq_gts: torch.FloatTensor, seq_preds: list[torch.FloatTensor], 
            best_seq_preds: list[torch.FloatTensor], psnrs: torch.FloatTensor, 
            num_cond: int, num_eval: int, num_samples: int, 
            path: str, duration: float) -> None: 
        for i in range(psnrs.shape[0]): 
            gifs = [[] for _ in range(num_eval)]
            texts = [[] for _ in range(num_eval)]
            
            mean_psnr = np.mean(psnrs[i], 1)
            mean_psnr_idx = np.argsort(mean_psnr)
            rands = [np.random.randint(num_samples) for _ in range(num_samples)]
            for frame_idx in range(num_eval): 
                # Ground-truth.
                gifs[frame_idx].append(
                    self.__add_borders(seq_gts[i, frame_idx], 'green'))
                texts[frame_idx].append('Ground\ntruth')
                
                color = 'green' if frame_idx < num_cond else 'red'
                
                # Posterior.
                gifs[frame_idx].append(
                    self.__add_borders(seq_preds[frame_idx][i], color))
                texts[frame_idx].append('Approx.\nposterior')
                
                # Best.
                idx = mean_psnr_idx[-1]
                gifs[frame_idx].append(
                    self.__add_borders(best_seq_preds[idx][frame_idx][i], color))
                texts[frame_idx].append('Best PSNR')
                
                # Random.
                for sample in range(len(rands)): 
                    gifs[frame_idx].append(
                        self.__add_borders(best_seq_preds[rands[sample]][frame_idx][i], color))
                    texts[frame_idx].append(f'Random\nsample {sample + 1}')
                    
        images = []
        for tensor, text in zip(gifs, texts):
            img = self.__image_tensor([
                self.__draw_text_tensor(ti, texti)
                for ti, texti in zip(tensor, text)], padding=0)
            
            img = img.to('cpu')
            img = img.transpose(0, 1).transpose(1, 2).clamp(0, 1).numpy()
            images.append(img)
            
        imageio.mimsave(path, img_as_ubyte(images), duration=duration)
        
    def __add_borders(self, 
            x: torch.FloatTensor, color: str, padding=1) -> torch.FloatTensor: 
        nc = x.shape[0]
        w = x.shape[1]
        
        px = Variable(torch.zeros(3, w + 2 * padding + 30, w + 2 * padding))
        
        if color == 'red':
            px[0] = 0.7
        elif color == 'green':
            px[1] = 0.7
            
        if nc == 1:
            for c in range(3):
                px[c, padding: w + padding, padding: w + padding] = x
        else:
            px[:, padding: w + padding, padding: w + padding] = x
            
        return px
    
    def __image_tensor(self, inputs: torch.FloatTensor, padding=1):
        # if this is a list of lists, unpack them all and grid them up
        if self.__is_sequence(inputs[0]) or (hasattr(inputs, 'dim') and inputs.dim() > 4):
            images = [self.__image_tensor(x) for x in inputs]
            if images[0].dim() == 3:
                c_dim = images[0].size(0)
                x_dim = images[0].size(1)
                y_dim = images[0].size(2)
            else:
                c_dim = 1
                x_dim = images[0].size(0)
                y_dim = images[0].size(1)

            result = torch.ones(c_dim,
                x_dim * len(images) + padding * (len(images) - 1),
                y_dim)  
            for i, image in enumerate(images):
                result[:, i * x_dim + i * padding:
                    (i + 1) * x_dim + i * padding, :].copy_(image)

            return result
        # if this is just a list, make a stacked image
        else:
            images = [x.data if isinstance(x, Variable) else x
                    for x in inputs]
            # print(images)
            if images[0].dim() == 3:
                c_dim = images[0].size(0)
                x_dim = images[0].size(1)
                y_dim = images[0].size(2)
            else:
                c_dim = 1
                x_dim = images[0].size(0)
                y_dim = images[0].size(1)

            result = torch.ones(c_dim, x_dim,
                y_dim * len(images) + padding * (len(images) - 1))
            for i, image in enumerate(images):
                result[:, :, i * y_dim + i * padding:
                    (i + 1) * y_dim + i * padding].copy_(image)
            return result
    
    def __is_sequence(self, arg):
        return (not hasattr(arg, 'strip') and
            not type(arg) is np.ndarray and
            not hasattr(arg, 'dot') and
            (hasattr(arg, '__getitem__') or
            hasattr(arg, '__iter__')))
        
    def __draw_text_tensor(self, tensor, text):
        np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
        pil = Image.fromarray(np.uint8(np_x * 255))
        
        draw = ImageDraw.Draw(pil)
        draw.text((4, 64), text, (0, 0, 0))
        img = np.asarray(pil)
        
        return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)
        
utils = __Utils()

__all__ = ['utils']
