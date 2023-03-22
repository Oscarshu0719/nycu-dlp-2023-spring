from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class __Utils(object):
    _instance = None 
    def __new__(cls): 
        if cls._instance is None: 
            cls._instance = super().__new__(cls) 
        return cls._instance
    
    def get_timestamp(self, timestamp=datetime.now()) -> str:
        return timestamp.strftime('%y%m%d_%H%M%S')
    
    def show_result(self, 
            x: np.ndarray, y: np.ndarray, pred_y: np.ndarray, is_saving=False, path=Union[str, None]) -> None: 
        """
        Args:
            x (np.ndarray): ground-truth inputs.
            y (np.ndarray): ground-truth labels.
            pred_y (np.ndarray): predicted labels.
            is_saving (bool, optional): determine if the figure is saved. Defaults to False.
            path (str or None, optional): path to save the image.
        """
        
        """
        Note: 
            Commented code below is original code given by TA, 
            but the created figure doesn't share axis.
        """
        # plt.subplot(1, 2, 1)
        # plt.title('Ground truth', fontsize=18)
        
        # for i in range(x.shape[0]): 
        #     if y[i] == 0:
        #         plt.plot(x[i][0], x[i][1], 'ro')
        #     else: 
        #         plt.plot(x[i][0], x[i][1], 'bo')
                
        # plt.subplot(1, 2, 2)
        # plt.title('Predict result', fontsize=18)
        
        # for i in range(x.shape[0]): 
        #     if pred_y[i] == 0:
        #         plt.plot(x[i][0], x[i][1], 'ro')
        #     else: 
        #         plt.plot(x[i][0], x[i][1], 'bo')
                
        # if is_saving:
        #     plt.savefig('test.png', dpi=200)
        #     plt.close()
        # else:
        #     plt.show()
            
        _, (ax, ax2) = plt.subplots(ncols=2, sharey=True)
        ax.set_title('Ground truth', fontsize=18)
        ax2.set_title('Predicted result', fontsize=18)
        ax.set_box_aspect(1)
        ax2.set_box_aspect(1)
            
        for i in range(x.shape[0]): 
            if y[i] == 0:
                ax.plot(x[i][0], x[i][1], 'ro')
            else: 
                ax.plot(x[i][0], x[i][1], 'bo')
                
        for i in range(x.shape[0]): 
            if pred_y[i] == 0:
                ax2.plot(x[i][0], x[i][1], 'ro')
            else: 
                ax2.plot(x[i][0], x[i][1], 'bo')
                
        if is_saving:
            plt.savefig(path if path else 'output.png', dpi=200, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        else:
            plt.show()

utils = __Utils()

__all__ = ['utils']
