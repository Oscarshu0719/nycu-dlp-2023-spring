## Setup

- `gym==0.15.7`.
- `atari-py==0.2.9`.


## Troubleshooting

### `Exception: ROM is missing for breakout`

``` bash
wget http://www.atarimania.com/roms/Roms.rar && \
    mkdir -p ./content/ROM/ && \
    unrar e ./Roms.rar ./content/ROM/ && \
    python -m atari_py.import_roms ./content/ROM/ && \
    rm -r ./content/ && rm Roms.rar
```
