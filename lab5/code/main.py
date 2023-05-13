import argparse
import os
from pathlib import Path

from src.solver import Solver


def main(config: dict, is_logging=True) -> None: 
    if is_logging:
        print(f'{config}\n')

    dir_dataset = Path(config.dir_dataset)
    assert dir_dataset.is_dir(), \
        f"Dataset {dir_dataset} doesn't exist (option: {dir_dataset})."
        
    assert config.num_cond + config.num_pred <= 30 and config.num_eval <= 30, 'Sum number of conditioning frames and predicted frames during training should be <= 30, and number of predicted frames during evaluation should satisfy, too.'
    assert 0 <= config.tf_ratio and config.tf_ratio <= 1, 'Teacher-forcing ratio should be in [0, 1].'
    assert 0 <= config.tf_ratio_min and config.tf_ratio_min <= 1, 'Teacher-forcing ratio lower bound should be in [0, 1].'
    assert 0 <= config.tf_epoch_start_decay, 'Epoch that teacher-forcing start to decay should be positive integer.'
    assert 0 <= config.tf_decay_step and config.tf_decay_step <= 1, 'Decay step size of teacher-forcing should be in [0, 1].'
    
    solver = Solver(config)
    if config.mode == 'train':
        solver.train()
    else: # 'test'.
        solver.eval()
    
if __name__ == '__main__':
    # CUDA debugging.
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
        help='Mode. Options: "train" (default) or "test".')
    parser.add_argument('--gpu', type=lambda s: [int(item) for item in s.split(',')], default=[3],  
        help='GPUs.')

    """
    Data configs.
    """
    parser.add_argument('--batch_size', type=int, default=64, 
        help='Mini-batch size.')
    parser.add_argument('--num_workers', type=int, default=8, 
        help='Number of subprocesses to use for data loading.')
    
    """
    Model configs.
    """    
    parser.add_argument('--num_rnn_hidden', type=int, default=256, 
        help='Dimension of hidden layers of RNN.')
    parser.add_argument('--num_rnn_pred', type=int, default=2, 
        help='Number of prediction layers of RNN.')
    parser.add_argument('--num_rnn_post', type=int, default=1, 
        help='Number of posterior layers of RNN.')
    parser.add_argument('--a_dim', type=int, default=7, 
        help='Dimension of action latent code.')
    parser.add_argument('--z_dim', type=int, default=64, 
        help='Dimension of RNN input latent code.')
    parser.add_argument('--g_dim', type=int, default=128, 
        help='Dimension of RNN output.')
    parser.add_argument('--is_last_frame_skipped', type=lambda x: x == 'True', default=False, 
        help='Determine if skip connections go between frame t and frame (t + t) rather than last ground truth frame.')

    """
    Training configs.
    """
    parser.add_argument('--num_epochs', type=int, default=100,  
        help='Number of epochs for training.')
    parser.add_argument('--num_iters', type=int, default=600,  
        help='Number of iterations of an epoch for training.')
    parser.add_argument('--lr', type=float, default=2e-3, 
        help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, 
        help='Random seed.')
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], 
        help='Optimizer. Options: "adam" (default), "adamw", or "sgd.')
    
    parser.add_argument('--num_cond', type=int, default=2, 
        help='Number of frames for conditioning.')
    parser.add_argument('--num_pred', type=int, default=10, 
        help='Number of frames to predict during training.')
    parser.add_argument('--num_eval', type=int, default=12, 
        help='Number of frames to predict during evaluation.')
    
    # Teacher-forcing.
    parser.add_argument('--tf_ratio', type=float, default=1.0, 
        help='Teacher-forcing ratio in [0, 1].')
    parser.add_argument('--tf_ratio_min', type=float, default=0, 
        help='Lower bound of teacher-forcing ratio in [0, 1].')
    parser.add_argument('--tf_epoch_start_decay', type=int, default=0, 
        help='Epoch that teacher-forcing start to decay.')
    parser.add_argument('--tf_decay_step', type=float, default=0, 
        help='Decay step size of teacher-forcing in [0, 1].')
    
    # KL loss.
    parser.add_argument('--kl_anneal_mode', type=str, default='mono', choices=['mono', 'cyc'],  
        help='KL annealing mode. Options: "mono" (default, monotonic) or "cyc" (cyclical).')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.5, 
        help='Decay ratio of KL annealing.')
    parser.add_argument('--kl_anneal_cycle', type=int, default=3, 
        help='Number of cycles for KL annealing during training under cyclical mode.')
    parser.add_argument('--kl_beta_min', type=float, default=1e-4, 
        help='Weight of KL prior loss (beta).')
    
    """
    Testing configs.
    """
    parser.add_argument('--resume_ckpt', type=str, default='', 
        help='Checkpoint to load (only for evaluation).')
    
    """
    Paths.
    """
    parser.add_argument('--dir_dataset', type=str, default='./datasets/', 
        help='Directory of dataset.')
    parser.add_argument('--dir_output', type=str, default='./output/', 
        help='Directory of outputs.')
    parser.add_argument('--dir_model', type=str, default='./models/', 
        help='Directory of model weights.')
    parser.add_argument('--file_record', type=str, default='record.txt', 
        help='File to store training loggings.')

    """
    Loggings and savings.
    """
    parser.add_argument('--epoch_check_psnr', type=int, default=5, 
        help='Number of epochs for checking PSNR.')
    parser.add_argument('--epoch_plot_pred', type=int, default=20, 
        help='Number of epochs for ploting prediction.')

    main(parser.parse_args(), is_logging=True)
