import numpy as np

from ..layers.linear import Linear


class Adam():
    def __init__(self, layers: list, lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m = []
        self.v = []
        for i in range(len(layers) - 1):
            W = np.zeros((layers[i + 1], layers[i]))
            b = np.zeros((1, layers[i + 1]))
            
            self.m.append(Linear(W, b))
            self.v.append(Linear(W, b))
        
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 1

    def update(self, layers: list[Linear], grads: list[Linear]) -> list[Linear]:
        new_layers = []
        for layer, grad, m, v in zip(layers, grads, self.m, self.v): 
            m.W = self.beta1 * m.W + (1 - self.beta1) * grad.W
            m.b = self.beta1 * m.b + (1 - self.beta1) * grad.b
            
            v.W = self.beta2 * v.W + (1 - self.beta2) * (grad.W ** 2)
            v.b = self.beta2 * v.b + (1 - self.beta2) * (grad.b ** 2)
            
            m_W_corr = m.W / (1 - self.beta1 ** self.t)
            m_b_corr = m.b / (1 - self.beta1 ** self.t)
            v_W_corr = v.W / (1 - self.beta2 ** self.t)
            v_b_corr = v.b / (1 - self.beta2 ** self.t)
            
            new_W = layer.W - self.lr * (m_W_corr / (np.sqrt(v_W_corr) + self.epsilon))
            new_b = layer.b - self.lr * (m_b_corr / (np.sqrt(v_b_corr) + self.epsilon))
            
            new_layers.append(Linear(new_W, new_b))
        
        self.t += 1
        
        return new_layers
