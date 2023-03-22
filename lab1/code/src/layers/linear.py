import numpy as np


class Linear(object):      
    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        self._W = W
        self._b = b
        
    @property
    def W(self) -> np.ndarray:
        return self._W
    
    @property
    def b(self) -> np.ndarray:
        return self._b
    
    @W.setter
    def W(self, W: np.ndarray) -> None:
        self._W = W
        
    @b.setter
    def b(self, b: np.ndarray) -> None:
        self._b = b
        
    def __str__(self) -> str:
        return f'<Linear W: {self._W.shape} b: {self._b.shape}>'
    