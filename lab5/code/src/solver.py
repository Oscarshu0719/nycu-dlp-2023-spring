import os
import numpy as np
from pathlib import Path
import random
import torch
from tqdm.auto import tqdm as tqdm

from src.dataloader import get_data_loaders
from src.kl_annealing import KlAnnealing
from src.models.lstm import GaussianLstm, Lstm
from src.models.vgg import VggDecoder, VggEncoder
from src.utils import utils


class Solver(object):
    def __init__(self, config: dict) -> None:
        self.timestamp = utils.get_timestamp()
        self.device = torch.device(f'cuda:{config.gpu[0]}' if torch.cuda.is_available() else 'cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, config.gpu)))
        
        print(f'***\nCurrently using {self.device}...\n***\n')
        
        """
        Model configs.
        """
        self.z_dim = config.z_dim
        self.is_last_frame_skipped = config.is_last_frame_skipped
        
        """
        Data configs.
        """
        self.batch_size = config.batch_size
        
        """
        Training configs.
        """
        self.num_epochs = config.num_epochs
        self.num_iters = config.num_iters
        
        """
        Teacher-forcing configs.
        """
        self.tf_ratio = config.tf_ratio
        self.tf_ratio_min = config.tf_ratio_min
        self.tf_epoch_start_decay = config.tf_epoch_start_decay
        self.tf_decay_step = config.tf_decay_step
        
        """
        RNN configs.
        """
        self.num_cond = config.num_cond
        self.num_pred = config.num_pred
        self.num_eval = config.num_eval
        
        """
        Paths.
        """
        self.file_record = config.file_record
        
        """
        Loggings and savings.
        """
        self.epoch_check_psnr = config.epoch_check_psnr
        self.epoch_plot_pred = config.epoch_plot_pred
        
        self.mkdirs(config)
        
        # Fix seed if not None.
        if config.seed != None:
            self.fix_seed(config.seed)
            
        # Get data loaders.
        [self.train_loader, self.valid_loader, self.test_loader] = get_data_loaders(
            config.dir_dataset, config.num_cond, config.num_pred, config.num_eval, 
            batch_size=config.batch_size, num_workers=config.num_workers)
        
        self.build_model(config)
        
    def train(self) -> None:
        train_iter = iter(self.train_loader)
        valid_iter = iter(self.valid_loader)
        len_seq = self.num_cond + self.num_pred
        
        best_psnr = 0
        for epoch in tqdm(range(self.epoch_start, self.epoch_start + self.num_epochs)): 
            # Training.
            self.predictor.train()
            self.posterior.train()
            self.encoder.train()
            self.decoder.train()
            
            epoch_loss = 0
            epoch_mse = 0
            epoch_kld = 0
            beta = self.kl_annealing.weight
            
            for _ in range(self.num_iters):
                try:
                    seq, cond = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    seq, cond = next(train_iter)
                finally:
                    # [b, len_seq, 3, 64, 64], [b, len_seq, a], 
                    # e,g., [128, 12, 3, 64, 64], [128, 12, 7].
                    seq = seq.to(self.device)
                    cond = cond.to(self.device)
                    
                self.predictor.zero_grad()
                self.posterior.zero_grad()
                self.encoder.zero_grad()
                self.decoder.zero_grad()
                
                # Init hidden state.
                self.predictor.hidden = self.predictor.init_hidden()
                self.posterior.hidden = self.posterior.init_hidden()
                
                kld = 0
                mse = 0
                use_teacher_forcing = True if random.random() < self.tf_ratio else False
                # Init as first frame.
                seq_pred = seq[:, 0]
                for frame_idx in range(len_seq - 1): 
                    input_prev = seq[:, frame_idx]
                    input_next = seq[:, frame_idx + 1]
                    cond_prev = cond[:, frame_idx]
                    
                    if use_teacher_forcing: 
                        h_prev = self.encoder(input_prev)
                    else:
                        h_prev = self.encoder(seq_pred)
                    
                    # h_prev: [b, g], e.g. [128, 128].
                    # skip: list, [4, 12].
                    if self.is_last_frame_skipped or frame_idx + 1 < self.num_cond:
                        h_prev, skip = h_prev
                    else:
                        h_prev = h_prev[0]
                        
                    # [b, g], e.g. [128, 128].
                    h_next = self.encoder(input_next)[0]
                    
                    # z, mu, logvar: [b, z], e.g. [128, 64].
                    z, mu, logvar = self.posterior(h_next)
                    
                    # [condition, hidden, z]: [b, a + g + z], e,g., [128, 7 + 128 + 64].
                    input_pred = torch.cat((cond_prev, h_prev, z), dim=1)
                    # [b, g], e.g., [128, 128].
                    h_pred = self.predictor(input_pred)
                    
                    # [b, 3, 64, 64], e.g., [128, 3, 64, 64].
                    seq_pred = self.decoder([h_pred, skip])
                                        
                    kld += utils.get_kld(mu, logvar, self.batch_size)
                    mse += utils.get_mse(seq_pred, input_next)
                
                loss = mse + beta * kld
                
                loss.backward()
                self.optim.step()
                
                epoch_loss += loss.detach().cpu().numpy() / len_seq
                epoch_mse += mse.detach().cpu().numpy() / len_seq
                epoch_kld += kld.detach().cpu().numpy() / len_seq
                
            self.tf_ratios.append(self.tf_ratio)
            self.kl_weights.append(beta)    
            self.loss_list.append(epoch_loss / self.num_epochs)
                
            # Update KL annealing epoch.
            self.kl_annealing.update()
            
            # Update Teacher-forcing ratio.
            if epoch >= self.tf_epoch_start_decay: 
                self.tf_ratio = max(
                    1 - self.tf_decay_step * (epoch - self.tf_epoch_start_decay), 
                    self.tf_ratio_min)
            
            epoch_mean_loss = epoch_loss / self.num_iters
            epoch_mean_mse = epoch_mse / self.num_iters
            epoch_mean_kld = epoch_kld / self.num_iters
            with open(Path(self.dir_output, self.file_record), 'a') as file:
                file.write(f'Epoch {epoch + 1:03d}: loss {epoch_mean_loss:.5f}, mse {epoch_mean_mse:.5f}, kld {epoch_mean_kld:.5f}\n')
                
            # Validation.
            self.predictor.eval()
            self.posterior.eval()
            self.encoder.eval()
            self.decoder.eval()
            
            if (epoch + 1) % self.epoch_check_psnr == 0:
                psnr_list = []
                for _ in range(len(self.valid_loader.dataset) // self.batch_size): 
                    try:
                        valid_seq, valid_cond = next(valid_iter)
                    except StopIteration:
                        valid_iter = iter(self.valid_loader)
                        valid_seq, valid_cond = next(valid_iter)
                    finally:
                        valid_seq = valid_seq.to(self.device)
                        valid_cond = valid_cond.to(self.device)
                    
                    pred_seq = self.pred(valid_seq, valid_cond)
                    psnr = utils.finn_eval_seq(valid_seq[:, self.num_cond: ], pred_seq[:, self.num_cond: ])
                    psnr_list.append(psnr)
                    
                mean_psnr = np.mean(np.concatenate(psnr_list))
                self.psnr_list.append(mean_psnr)
                with open(Path(self.dir_output, self.file_record), 'a') as file:
                    file.write(f'*** Validation PSNR: {mean_psnr:.5f}\n')
                    
                if mean_psnr > best_psnr: 
                    best_psnr = mean_psnr
                    print(f'* Current best training PSNR: {best_psnr:.5f}.')
                    self.save_model(epoch + 1)

            if (epoch + 1) % self.epoch_plot_pred == 0: 
                try:
                    valid_seq, valid_cond = next(valid_iter)
                except StopIteration:
                    valid_iter = iter(self.valid_loader)
                    valid_seq, valid_cond = next(valid_iter)
                finally:
                    valid_seq = valid_seq.to(self.device)
                    valid_cond = valid_cond.to(self.device)
                    
                pred_seq = self.pred(valid_seq, valid_cond)
                utils.plot_pred(valid_seq, pred_seq, epoch + 1, self.dir_output)
                
        utils.plot_result(
            self.loss_list, self.psnr_list, self.kl_weights, self.tf_ratios, 
            self.epoch_start, self.num_epochs, self.epoch_check_psnr, self.dir_output)
        
    def pred(self, seq: torch.FloatTensor, cond: torch.FloatTensor) -> torch.FloatTensor:
        # Init hidden state.
        self.predictor.hidden = self.predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        
        with torch.no_grad():
            seq_pred = seq[:, 0]
            seq_preds = torch.Tensor(seq[:, 0].unsqueeze(1)).to(self.device)
            for frame_idx in range(self.num_cond + self.num_pred - 1): 
                input_next = seq[:, frame_idx + 1]
                cond_prev = cond[:, frame_idx]
                
                h_prev = self.encoder(seq_pred)
                # h_prev: [b, g], e.g. [128, 128].
                # skip: list, [4, 12].
                if self.is_last_frame_skipped or frame_idx + 1 < self.num_cond: 
                    h_prev, skip = h_prev
                else: 
                    h_prev = h_prev[0]
                
                # [b, g], e.g. [128, 128].
                h_next = self.encoder(input_next)[0]
                
                # [b, z], e.g. [128, 64].
                z, _, _ = self.posterior(h_next)
                
                # [condition, hidden, z]: [b, a + g + z], e,g., [128, 7 + 128 + 64].
                input_pred = torch.cat((cond_prev, h_prev, z), dim=1)
                # [b, g], e.g., [128, 128].
                h_pred = self.predictor(input_pred)
                
                # [b, 3, 64, 64], e.g., [128, 3, 64, 64].
                seq_pred = self.decoder([h_pred, skip])
                
                # [b, len_seq, 3, 64, 64], e.g., [128, 12, 3, 64, 64].
                seq_preds = torch.cat((seq_preds, seq_pred.unsqueeze(1)), dim=1)
        
        return seq_preds
        
    def eval(self) -> None: 
        self.predictor.eval()
        self.posterior.eval()
        self.encoder.eval()
        self.decoder.eval()
        
        test_iter = iter(self.test_loader)
        with torch.no_grad(): 
            psnr_list = []
            for _ in tqdm(range(len(self.test_loader.dataset) // self.batch_size)): 
                try:
                    test_seq, test_cond = next(test_iter)
                except StopIteration:
                    test_iter = iter(self.test_loader)
                    test_seq, test_cond = next(test_iter)
                finally:
                    test_seq = test_seq.to(self.device)
                    test_cond = test_cond.to(self.device)
                    
                pred_seq = self.pred(test_seq, test_cond)
                psnr = utils.finn_eval_seq(test_seq[:, self.num_cond: ], pred_seq[:, self.num_cond: ])
                psnr_list.append(psnr)
                
            mean_psnr = np.mean(np.concatenate(psnr_list))
            self.psnr_list.append(mean_psnr)
            
            msg = f'*** Testing PSNR: {mean_psnr:.5f}'
            with open(Path(self.dir_output, self.file_record), 'a') as file:
                file.write(f'{msg}\n')
            print(msg)
                
            try:
                test_seq, test_cond = next(test_iter)
            except StopIteration:
                test_iter = iter(self.test_loader)
                test_seq, test_cond = next(test_iter)
            finally:
                test_seq = test_seq.to(self.device)
                test_cond = test_cond.to(self.device)
            
            self.eval_and_plot(test_seq, test_cond)
                
    def eval_and_plot(self, seq: torch.FloatTensor, cond: torch.FloatTensor) -> None: 
        # Init hidden state.
        self.predictor.hidden = self.predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        
        seq_pred = seq[:, 0]
        seq_preds = [seq[:, 0]]
        for frame_idx in range(self.num_eval - 1): 
            input_next = seq[:, frame_idx + 1]
            cond_prev = cond[:, frame_idx]
            
            h_prev = self.encoder(seq_pred)
            # h_prev: [b, g], e.g. [128, 128].
            # skip: list, [4, 12].
            if self.is_last_frame_skipped or frame_idx + 1 < self.num_cond:
                h_prev, skip = h_prev
            else:
                h_prev, _ = h_prev
            h_prev = h_prev.detach()
            
            # [b, g], e.g. [128, 128].
            h_next = self.encoder(input_next)[0].detach()
            
            # [b, z], e.g. [128, 64].
            _, mu, _ = self.posterior(h_next)
            
            # [condition, hidden, z]: [b, a + g + z], e,g., [128, 7 + 128 + 64].
            input_pred = torch.cat((cond_prev, h_prev, mu), dim=1)
            if frame_idx + 1 < self.num_cond: 
                self.predictor(input_pred)
                seq_preds.append(input_next)
                seq_pred = input_next
            else: 
                h_pred = self.predictor(input_pred).detach()
                seq_pred = self.decoder([h_pred, skip]).detach()
                seq_preds.append(seq_pred)
        
        num_samples = 3
        psnrs = np.zeros((self.batch_size, num_samples, self.num_pred))
        best_seq_preds = []
        for sample in tqdm(range(num_samples)): 
            seq_gens = []
            seq_gts = []
            
            # Init hidden state.
            self.predictor.hidden = self.predictor.init_hidden()
            self.posterior.hidden = self.posterior.init_hidden()

            seq_pred = seq[:, 0]
            best_seq_preds.append([seq[:, 0]])
            for frame_idx in range(self.num_eval - 1): 
                input_next = seq[:, frame_idx + 1]
                cond_prev = cond[:, frame_idx]
                
                h_prev = self.encoder(seq_pred)
                # h_prev: [b, g], e.g. [128, 128].
                # skip: list, [4, 12].
                if self.is_last_frame_skipped or frame_idx + 1 < self.num_cond: 
                    h_prev, skip = h_prev
                else: 
                    h_prev, _ = h_prev
                h_prev = h_prev.detach()
                
                # h_next: [b, g], e.g. [128, 128].
                # mu: [b, z], e.g. [128, 64].
                if frame_idx + 1 < self.num_cond:
                    h_next = self.encoder(input_next)[0].detach()
                    _, mu, _ = self.posterior(h_next)
                else:
                    mu = torch.randn(self.batch_size, self.z_dim).to(self.device)
                    
                # [condition, hidden, z]: [b, a + g + z], e,g., [128, 7 + 128 + 64].
                input_pred = torch.cat((cond_prev, h_prev, mu), dim=1)
                if frame_idx + 1 < self.num_cond:
                    self.predictor(input_pred)
                    best_seq_preds[sample].append(input_next)
                    seq_pred = input_next
                else:
                    h_pred = self.predictor(input_pred).detach()
                    seq_pred = self.decoder([h_pred, skip]).detach()
                
                    seq_gens.append(seq_pred)
                    seq_gts.append(input_next)
                    best_seq_preds[sample].append(seq_pred)
            
            psnrs[:, sample, :] = utils.finn_eval_seq(seq_gts, seq_gens)
        
        utils.ssim_and_plot(
            seq, seq_preds, best_seq_preds, psnrs, 
            self.num_cond, self.num_eval, num_samples, 
            Path(self.dir_output, 'test.gif'), 0.25)        
                
    def build_model(self, config: dict) -> None:
        self.epoch_start = 0
        
        # Models.
        if config.resume_ckpt.strip() != '':
            self.load_model(config.resume_ckpt.strip())
        else: 
            self.predictor = Lstm(
                config.a_dim + config.g_dim + config.z_dim, config.g_dim, 
                config.num_rnn_hidden, config.num_rnn_pred, 
                config.batch_size, self.device)
            self.posterior = GaussianLstm(
                config.g_dim, config.z_dim, 
                config.num_rnn_hidden, config.num_rnn_post, 
                config.batch_size, self.device)
            self.encoder = VggEncoder(config.g_dim)
            self.decoder = VggDecoder(config.g_dim)
            
            self.predictor.apply(utils.init_weights)
            self.posterior.apply(utils.init_weights)
            self.encoder.apply(utils.init_weights)
            self.decoder.apply(utils.init_weights)
            
        self.predictor = self.predictor.to(self.device)
        self.posterior = self.posterior.to(self.device)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        # Optimizers.
        params = list(self.predictor.parameters()) + list(self.posterior.parameters()) + \
            list(self.encoder.parameters()) + list(self.decoder.parameters())
        if config.optim == 'adam': 
            self.optim = torch.optim.Adam(params, 
                lr=config.lr, betas=(0.9, 0.999))
        elif config.optim == 'adamw': 
            self.optim = torch.optim.AdamW(params, 
                lr=config.lr, betas=(0.9, 0.999))
        else: # 'sgd'.
            self.optim = torch.optim.SGD(params, 
                lr=config.lr, momentum=0.9)
            
        # KL annealing.
        self.kl_annealing = KlAnnealing(
            config.kl_anneal_ratio, config.kl_anneal_cycle, 
            config.num_epochs, config.kl_beta_min, 
            config.kl_anneal_mode)
        
        # Loggings.
        self.tf_ratios = []
        self.kl_weights = []
        self.loss_list = []
        self.psnr_list = []
    
    def load_model(self, path: str) -> None:
        path_ckpt = Path(path)
        try: 
            pth = torch.load(path_ckpt, map_location='cpu')
            self.predictor = pth['pred']
            self.posterior = pth['post']
            self.encoder = pth['enc']
            self.decoder = pth['dec']
            self.epoch_start = pth['epoch']
            
            # Update GPUs.
            self.predictor.device = self.device
            self.posterior.device = self.device
            
            msg = f'Finished loading checkpoint from {path_ckpt}. Starting from epoch {self.epoch_start + 1}...'
            with open(Path(self.dir_output, self.file_record), 'a') as file:
                file.write(f'{msg}\n{"-" * 49}\n')
            print(msg)
        except Exception as msg:
            print(f'Failed to load checkpoint from {path_ckpt}. Message: \n{msg}')
            
    def save_model(self, epoch: int) -> None:
        # ONLY save best model.
        torch.save({
            'pred': self.predictor, 
            'post': self.posterior, 
            'enc': self.encoder, 
            'dec': self.decoder, 
            'epoch': epoch
        }, Path(self.dir_model, f'{self.timestamp}.pth'))
            
    def mkdirs(self, config: dict) -> None:
        path = (f'{self.timestamp}_rnn-d{config.num_rnn_hidden}post{config.num_rnn_post}pred{config.num_rnn_pred}'
                f'g{config.g_dim}z{config.z_dim}skip{"Y" if config.is_last_frame_skipped else "X"}_'
                f'batch{config.batch_size}lr{config.lr}{config.optim}_'
                f'cond{config.num_cond}pred{config.num_pred}_'
                f'{config.kl_anneal_mode}-ratio{config.kl_anneal_ratio}cycle{config.kl_anneal_cycle}min{config.kl_beta_min}')
        
        if config.mode == 'train': # 'train'.
            self.dir_model = Path(config.dir_model, path)
            
            dir_model = Path(self.dir_model)
            if not dir_model.is_dir(): 
                dir_model.mkdir(parents=True, exist_ok=True)
                
        # 'train' or 'test'.
        self.dir_output = Path(config.dir_output, path)
            
        dir_output = Path(self.dir_output)
        if not dir_output.is_dir(): 
            dir_output.mkdir(parents=True, exist_ok=True)
            
        with open(Path(self.dir_output, self.file_record), 'a') as file:
            file.write(f'{path}\n{"-" * 49}\n')
            
    def fix_seed(self, seed) -> None:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True