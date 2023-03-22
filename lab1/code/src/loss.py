import numpy as np


class __Loss(object):
    _instance = None 
    def __new__(cls): 
        if cls._instance is None: 
            cls._instance = super().__new__(cls) 
        return cls._instance
    
    def cross_entropy(self, real_y: np.ndarray, pred_y: np.ndarray) -> np.ndarray: 
        dim = real_y.ndim
        assert dim == pred_y.ndim, f'Number of dimension of real and predicted labels should be the same. {dim} expected, but {pred_y.ndim} found.'
        for i in range(dim):
            assert real_y.shape[i] == pred_y.shape[i], f'Dimension {i + 1} of real and predicted labels should be the same. {real_y.shape[i]} expected, but {pred_y.shape[i]} found.'
        
        EPS = 1e-3
        loss = (1. / real_y.shape[0]) * (
            -np.dot(real_y.T, np.log(pred_y + EPS)) - np.dot(1 - real_y.T, np.log(1 - pred_y + EPS)))
        
        return loss
    
loss = __Loss()

__all__ = ['loss']