import argparse
import os
from pathlib import Path

from src.solver import Solver


def main(config: dict, is_logging=True) -> None:
    if is_logging:
        print(f'{config}\n')
    
    dir_dataset = Path(config.dir_dataset)
    assert dir_dataset.is_dir(), \
        f'Directory of dataset {dir_dataset} is NOT a directory'
        
    Solver(config)

if __name__ == '__main__':
    # For CUDA debugging.
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
        help='Mode. Options: "train" (default) or "test".')
    parser.add_argument('--gpu', type=lambda s: [int(item) for item in s.split(',')], default=[0],  
        help='GPUs.')

    """
    Model configs.
    """
    parser.add_argument('--resume_ckpt', type=str, default='',  
        help='Checkpoint to resume (root dir: *dir_model*).')
        
    """
    Data configs.
    """
    parser.add_argument('--batch_size', type=int, default=128, 
        help='Mini-batch size.')
    parser.add_argument('--num_workers', type=int, default=8, 
        help='Number of subprocesses to use for data loading.')

    """
    Directories.
    """
    parser.add_argument('--dir_dataset', type=str, default='./datasets/', 
        help='Directory of dataset.')
    parser.add_argument('--dir_model', type=str, default='./models/', 
        help='Directory of saved models.')
    parser.add_argument('--dir_writer', type=str, default='./runs/', 
        help='Directory of summary writer.')
    parser.add_argument('--dir_output', type=str, default='./output/', 
        help='Directory of generated tokens.')

    """
    Loggings and savings.
    """
    parser.add_argument('--epoch_model_saving', type=int, default=10, 
        help='Number of epochs for model saving.')
    parser.add_argument('--epoch_img_saving', type=int, default=10, 
        help='Number of epochs for image saving.')

    """
    Training configs.
    """
    parser.add_argument('--train_num_epochs', type=int, default=100,  
        help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=1e-4, 
        help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, 
        help='Random seed.')
    parser.add_argument('--optim', type=str, default='adam', choices=['adam'], 
        help='Optimizer. Options: "adam" (default).')
    
    main(parser.parse_args(), is_logging=True)
