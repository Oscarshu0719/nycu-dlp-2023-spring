from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


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
    
    def plot_result(self, 
            vals: list, num_epochs: int, path: str, 
            title: str, labels: list, legend_loc='lower right') -> None:
        assert len(vals) == len(labels), f'Length of values and labels should be the same ({len(vals)} and {len(labels)} are found).'
        
        plt.figure(figsize=(15, 15))
        
        lin = np.linspace(1, num_epochs, num_epochs).astype(int)
        for val, label in zip(vals, labels): 
            plt.plot(lin, val, label=label)
        
        plt.xlabel('Epoch', fontsize='18') 
        plt.ylabel(title, fontsize='18')
        
        if title == 'Accuracy': 
            yticks = [0.5, 0.6, 0.7, 0.8, 0.85, 0.87, 0.9, 0.95, 1.0]
            axes = plt.gca()
            axes.grid(axis='y')
            axes.set_yticks(yticks)
        
        plt.legend(loc=legend_loc, fontsize='18', shadow=True)
        
        plt.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.1) 
        plt.close()
        
        print(f'[INFO]: Finish saving {title} to {path}...')

utils = __Utils()

__all__ = ['utils']
