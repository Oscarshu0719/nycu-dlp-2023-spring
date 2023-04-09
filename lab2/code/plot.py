import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def plot(x: np.ndarray, y: np.ndarray, path: str, unit=1000) -> None: 
    assert x.ndim == 1 and y.ndim == 1, f'Number of dimensions of `x` and `y` should be 1, but {x.ndim} and {y.ndim} are found.'
    assert x.shape[0] == y.shape[0], f'Dimension of `x` and `y` should be the same, but {x.shape[0]} and {y.shape[0]} are found.'
    
    # Interpolation.
    spline = make_interp_spline(x, y)
    linspace_x = np.linspace(x.min(), x.max(), 500)
    linspace_y = spline(linspace_x)
    
    plt.figure(figsize=(15, 15))
    plt.plot(linspace_x, linspace_y)
    
    # Y ticks.
    yticks = [i * 10000 for i in range(int(np.max(y)) // 10000 + 1)]
    yticks.append(np.max(y)) # Annotate max score.
    yticks.insert(0, np.min(y)) # Annotate min score.
    axes = plt.gca()
    axes.grid(axis='y')
    axes.set_yticks(yticks)

    plt.xlabel('Iterations', fontsize='18') 
    plt.ylabel('Score', fontsize='18')
    plt.title(f'Mean score of each {unit} iterations', fontsize='24')
    
    plt.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.1) 
    plt.close()
    
    print(f'[INFO]: Plot has been saved to {path}.')
    
def load_by_all(path: str) -> np.ndarray: 
    data = []
    with open(path, 'r') as file: 
        data = file.readline().split(',')
        data = [x for x in data if x != ''] # Remove empty data.
        
    return np.array(data, dtype=np.int64)

def get_means(data: np.ndarray, unit=1000) -> np.ndarray:
    assert data.ndim == 1, f'Number of dimensions of `data` should be 1, but {data.ndim} found.'
    
    numUnits = data.shape[0] // unit
    means = np.zeros(numUnits)
    indices = np.zeros(numUnits)
    
    for i in range(numUnits): 
        indices[i] = unit * (i + 1) - 1
        means[i] = np.mean(data[unit * i: unit * (i + 1)])
    
    return means, indices
    
if __name__ == '__main__': 
    PATH = ['./tmp/output/output.txt', './output/output.txt']
    UNIT = 1000
    
    data = load_by_all(PATH[0])[: 240000]
    data = np.concatenate((data, load_by_all(PATH[1])[: 110000]))
        
    means, idx = get_means(data, UNIT)
    plot(idx, means, 'result.png', UNIT)
    