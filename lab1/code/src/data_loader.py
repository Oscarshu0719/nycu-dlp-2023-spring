import numpy as np


class __DataLoader(object):
    _instance = None 
    def __new__(cls): 
        if cls._instance is None: 
            cls._instance = super().__new__(cls) 
        return cls._instance
    
    def generate_linear(self, n=100) -> tuple[np.ndarray, np.ndarray]: 
        pts = np.random.uniform(0, 1, (n, 2))
        inputs = []
        labels = []
        
        for pt in pts: 
            inputs.append([pt[0], pt[1]])
            
            if pt[0] > pt[1]:
                labels.append(0)
            else:
                labels.append(1)
                
        return np.array(inputs), np.array(labels).reshape(n, 1)

    def generate_XOR_easy(self) -> tuple[np.ndarray, np.ndarray]: 
        inputs = []
        labels = []
        
        for i in range(11): 
            inputs.append([0.1 * i, 0.1 * i])
            labels.append(0)
            
            if 0.1 * i == 0.5: 
                continue
            
            inputs.append([0.1 * i, 1 - 0.1 * i])
            labels.append(1)
            
        return np.array(inputs), np.array(labels).reshape(21, 1)
         
data_loader = __DataLoader()

__all__ = ['data_loader']
