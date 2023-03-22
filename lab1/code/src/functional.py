import numpy as np


class __Functional(object):
    _instance = None 
    def __new__(cls): 
        if cls._instance is None: 
            cls._instance = super().__new__(cls) 
        return cls._instance

    def relu(self, x: np.ndarray) -> np.ndarray: 
        return np.maximum(0, x)
    
    def deriv_relu(self, grad: np.ndarray, deriv: np.ndarray) -> np.ndarray: 
        new_grad = np.array(grad, copy=True)
        new_grad[deriv < 0] = 0
        
        return new_grad
    
    def leaky_relu(self, x: np.ndarray) -> np.ndarray: 
        new_x = np.array(x, copy=True)
        new_x[new_x < 0] = 0.01 * new_x[new_x < 0]
        
        return new_x
    
    def deriv_leaky_relu(self, grad: np.ndarray, deriv: np.ndarray) -> np.ndarray: 
        new_grad = np.array(grad, copy=True)
        new_grad[deriv < 0] = 0.01 * new_grad[deriv < 0]
        
        return new_grad
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray: 
        return 1 / (1 + np.exp(-x))
    
    def deriv_sigmoid(self, x: np.ndarray) -> np.ndarray: 
        sigmoid = self.sigmoid(x)
        return sigmoid * (1 - sigmoid)
        
func = __Functional()

__all__ = ['func']
