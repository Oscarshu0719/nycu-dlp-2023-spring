import numpy as np
from typing import Union

from .functional import func
from .layers.linear import Linear
from .optim.adam import Adam
from .optim.vanilla import Vanilla


class Model(object):
    def __init__(self, 
            layers: list, lr: float, activation: str, optimizer: str, seed: Union[int, None]=None) -> None:
        self.lr = lr
        self.name_activation = activation
        self.name_optim = optimizer
        self.seed = seed
        self.EPS = 1e-3
        
        if activation == 'relu':
            self.activation = func.relu
            self.deriv_activation = func.deriv_relu
        elif activation == 'sigmoid':
            self.activation = func.sigmoid
            self.deriv_activation = func.deriv_sigmoid
        else: # 'leaky_relu'.
            self.activation = func.leaky_relu
            self.deriv_activation = func.deriv_leaky_relu
            
        if optimizer == 'vanilla':
            self.optim = Vanilla(lr=self.lr)
        else: # 'adam'.
            self.optim = Adam(layers, lr=self.lr)
        
        self.init_weights(layers=layers, seed=seed)
    
    def init_weights(self, layers: list, seed: Union[int, None]) -> None: 
        if seed is not None: 
            np.random.seed(seed)
        
        self.layers = []
        for i in range(len(layers) - 1):
            # He normal initialization.
            dim_in = layers[i]
            dim_out = layers[i + 1]
            a = np.sqrt(6. / (dim_in + dim_out))
            std = np.sqrt(2. / ((1 + a ** 2) * dim_in))
            
            W = np.random.normal(loc=0., scale=std, size=[dim_out, dim_in])
            b = np.random.normal(loc=0., scale=std, size=[1, dim_out])
            
            self.layers.append(Linear(W, b))
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        assert inputs.shape[1] == self.layers[0].W.shape[1], f'Dimension of inputs and first layer should be the same. {self.layers[0].W.shape[1]} expected, but {inputs.shape[1]} found.'
        
        x = inputs
        self.cache = []
        for layer in self.layers[: -1]: 
            # Linear. (y = xW^T + b)
            tmp = np.dot(x, layer.W.T) + layer.b
            # Activation.
            x = self.activation(tmp)
            
            self.cache.append([tmp, x]) # [Zi, Ai].
            
        flatten_layer = self.layers[-1]
        x = np.dot(x, flatten_layer.W.T) + flatten_layer.b
        output = func.sigmoid(x)
        
        self.cache.append([x, output]) # [Zn, An].
        
        return output
        
    def backward(self, real_x: np.ndarray, real_y: np.ndarray) -> None:
        self.cache.reverse()
        num = real_y.shape[0]
        
        # dA3.
        grad = - (np.divide(real_y, self.cache[0][1] + self.EPS) - np.divide(1 - real_y, 1 - self.cache[0][1] + self.EPS))
        # dZ3.
        grad = grad * func.deriv_sigmoid(self.cache[0][0])
        # dW3.
        dW3 = np.dot(grad.T, self.cache[1][1]) / num
        # db3.
        db3 = np.sum(grad, axis=0, keepdims=True) / num
        
        # dA2.
        grad = np.dot(grad, self.layers[-1].W)
        # dZ2. (ReLU)
        grad = self.deriv_activation(grad, self.cache[1][0])
        # dW2.
        dW2 = np.dot(grad.T, self.cache[2][1]) / num
        # db2.
        db2 = np.sum(grad, axis=0, keepdims=True) / num
        
        # dA1.
        grad = np.dot(grad, self.layers[-2].W)
        # dZ1. (ReLU)
        grad = self.deriv_activation(grad, self.cache[2][0])
        # dW1.
        dW1 = np.dot(grad.T, real_x) / num
        # db1.
        db1 = np.sum(grad, axis=0, keepdims=True) / num
        
        grads = [
            Linear(dW1, db1),
            Linear(dW2, db2),
            Linear(dW3, db3)]
        
        self.layers = self.optim.update(self.layers, grads)
        
    def __str__(self) -> str:
        msg = [
            f'<Linear W: {layer.W.shape} b: {layer.b.shape}>'
            for layer in self.layers
        ]
        return (f'<{", ".join(msg)} (total {len(self.layers)} layer(s))'
                f', lr: {self.lr}, activation: {self.name_activation}'
                f', optimizer: {self.name_optim}, seed: {self.seed}>') 
            