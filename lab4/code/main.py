import argparse
import os
from pathlib import Path

from src.solver import Solver


def main(config: dict, is_logging=True) -> None: 
    if is_logging:
        print(f'{config}\n')

    dir_dataset = Path(config.dir_dataset)
    assert dir_dataset.exists(), \
        f"Dataset {dir_dataset} doesn't exist (option: {dir_dataset})."
    
    
    solver = Solver(config)
    if config.mode == 'train':
        solver.train_all()
    else: # 'test'.
        solver.eval_all()
    
if __name__ == '__main__':
    # For CUDA debugging.
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
        help='Mode. Options: "train" (default) or "test".')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'], 
        help='Model. Options: "resnet18" (default), or "resnet50".')
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
    Paths.
    """
    parser.add_argument('--dir_dataset', type=str, default='./datasets/', 
        help='Directory of dataset.')
    parser.add_argument('--dir_output', type=str, default='./output/', 
        help='Directory of outputs.')
    parser.add_argument('--dir_model', type=str, default='./models/', 
        help='Directory of model weights.')

    """
    Loggings and savings.
    """
    parser.add_argument('--epoch_log', type=int, default=1, 
        help='Number of epochs for logging.')
    parser.add_argument('--is_model_printing', type=lambda x: x == 'True', default=False, 
        help='Determine if model details are printed.')
    parser.add_argument('--is_plot_saving', type=lambda x: x == 'True', default=True,   
        help='Determine if plots of accuracy and loss are saved.')

    """
    Training configs.
    """
    parser.add_argument('--num_epochs', type=int, default=10,  
        help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=1e-3, 
        help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, 
        help='Random seed.')
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'], 
        help='Optimizer. Options: "sgd" (default), "adamw", or "adam.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, 
        help='Weight decay.')
    
    """
    Testing configs.
    """
    parser.add_argument('--resume_ckpt', type=str, default='', 
        help='Checkpoint to load (only for evaluation).')

    main(parser.parse_args(), is_logging=True)
