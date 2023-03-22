from ..layers.linear import Linear


class Vanilla(object):
    def __init__(self, lr: float) -> None:
        self.lr = lr
    
    def update(self, layers: list[Linear], grads: list[Linear]) -> list[Linear]:
        new_layers = []
        for layer, grad in zip(layers, grads): 
            new_W = layer.W - grad.W * self.lr
            new_b = layer.b - grad.b * self.lr
            
            new_layers.append(Linear(new_W, new_b))
            
        return new_layers
    