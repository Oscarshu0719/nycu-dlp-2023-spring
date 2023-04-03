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
    
    Solver(config)
    
if __name__ == '__main__':
    # For CUDA debugging.
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='eegnet', choices=['eegnet', 'deepconvnet'], 
        help='Model. Options: "eegnet" (EEGNet, default), or "deepconvnet" (DeepConvNet).')
    parser.add_argument('--gpu', type=lambda s: [int(item) for item in s.split(',')], default=[2],  
        help='GPUs.')

    """
    Data configs.
    """
    parser.add_argument('--batch_size', type=int, default=64, 
        help='Mini-batch size.')
    parser.add_argument('--num_workers', type=int, default=4, 
        help='Number of subprocesses to use for data loading.')

    """
    Paths.
    """
    parser.add_argument('--dir_dataset', type=str, default='./datasets/', 
        help='Directory of dataset.')
    parser.add_argument('--dir_output', type=str, default='./output/', 
        help='Directory of outputs.')
    parser.add_argument('--path_acc', type=str, default='./acc.txt', 
        help='Path of accuracy output file.')

    """
    Loggings and savings.
    """
    parser.add_argument('--epoch_log', type=int, default=10, 
        help='Number of epochs for logging.')
    parser.add_argument('--is_model_printing', type=lambda x: x == 'True', default=False, 
        help='Determine if model details are printed.')
    parser.add_argument('--is_plot_saving', type=lambda x: x == 'True', default=True,   
        help='Determine if plot the scatter of t-SNE output and save as an image.')

    """
    Training configs.
    """
    parser.add_argument('--num_epochs', type=int, default=300,  
        help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=1e-2, 
        help='Learning rate.')
    parser.add_argument('--seed', type=int, default=42, 
        help='Random seed.')
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], 
        help='Optimizer. Options: "adam" (default), "adamw", or "sgd" (learning rate is auto 100x).')
    parser.add_argument('--dropout', type=float, default=0.25, 
        help='Dropout.')

    main(parser.parse_args(), is_logging=True)
