from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import PIL
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
    
    def center_crop(self, img: PIL.Image): 
        width, height = img.size
        cropped = img.crop(((width - height) / 2, 0, (width + height) / 2, height))
        
        return cropped
    
    def plot_result(self, 
            vals: list, num_epochs: int, path: str, 
            title: str, labels: list, legend_loc='lower right') -> None:
        assert len(vals) == len(labels), f'Length of values and labels should be the same ({len(vals)} and {len(labels)} are found).'
        
        plt.figure(figsize=(15, 15))
        
        lin = np.linspace(1, num_epochs, num_epochs).astype(int)
        for val, label in zip(vals, labels): 
            plt.plot(lin, val, label=label)
        
        plt.xlabel('Epochs', fontsize='18') 
        plt.ylabel(title, fontsize='18')
        
        plt.legend(loc=legend_loc, fontsize='18', shadow=True)
        
        plt.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.1) 
        plt.close()
        
        print(f'[INFO]: Finish saving {title} to {path}...')
        
    def confusion_matrix(self, 
            true_label: np.ndarray, pred_label: np.ndarray, path: str, model_name: str) -> None:
        plt.figure(figsize=(15, 15))
        
        plt.title(f'Confusion matrix of {model_name}')
        plt.xlabel('Predicted label', fontsize='18')
        plt.ylabel('True label', fontsize='18') 
        
        cm = confusion_matrix(y_true=true_label, y_pred=pred_label, normalize='true')
        ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4]).plot(cmap=plt.cm.Blues)
        
        plt.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.1) 
        plt.close()
        
        print(f'[INFO]: Finish saving confusion matrix to {path}...')

utils = __Utils()

__all__ = ['utils']
