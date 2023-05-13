class KlAnnealing(object):
    def __init__(self, 
            ratio: float, cycle: int, num_epochs: int, beta_min: float, 
            mode='mono') -> None:
        self.cycle_num_epochs = num_epochs // cycle
        self.step = 1 / (self.cycle_num_epochs * ratio)
        
        self.beta_min = beta_min
        self.epoch = 0
        
        self.__method = self.__monotonic if mode == 'mono' else self.__cyclical
        
    def __monotonic(self) -> float:
        beta = min(self.beta_min + self.epoch * self.step, 1.0)
        
        return beta
    
    def __cyclical(self) -> float:
        beta = min(self.beta_min + (self.epoch % self.cycle_num_epochs) * self.step, 1.0)
            
        return beta
    
    def update(self) -> None:
        self.epoch += 1
    
    @property
    def weight(self) -> float:
        return self.__method()
    
if __name__ == '__main__': 
    import matplotlib.pyplot as plt
    import numpy as np
    
    def plot_result(
            vals: list, num_epochs: int, labels: list, 
            title: str, ylabel: str, path: str) -> None:
        assert len(vals) == len(labels), f'Length of values and labels should be the same ({len(vals)} and {len(labels)} are found).'
        
        plt.figure(figsize=(15, 15))
        
        lin = np.linspace(1, num_epochs, num_epochs).astype(int)
        for val, label in zip(vals, labels): 
            plt.plot(lin, val, label=label)
        
        plt.xlabel('Epochs', fontsize='18') 
        plt.ylabel(ylabel, fontsize='18')
        plt.title(title, fontsize='24')
        
        plt.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.1) 
        plt.close()
     
    num_epochs = 100
    mode = 'mono'
    kl = KlAnnealing(1.0, 3, num_epochs, 1e-4, mode)
    weights = []
    for i in range(num_epochs):
        weights.append(kl.weight)
        kl.update()
    
    title = 'Monotonic' if mode == 'mono' else 'Cyclical'
    plot_result(
        [weights], num_epochs, ['KL'], 
        f'KL Annealing weights (mode: {title})', 'Beta', f'./{mode}.png')
    