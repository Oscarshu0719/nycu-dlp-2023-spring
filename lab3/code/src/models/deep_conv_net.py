from functools import reduce
import numpy as np
import torch
from torch import nn


class DeepConvNet(nn.Module): 
    def __init__(self, act_name='relu', dropout=0.5) -> None: 
        super().__init__()
        
        if act_name == 'elu':
            self.act = nn.ELU()
        elif act_name == 'leakyrelu':
            self.act = nn.LeakyReLU()
        else: # 'relu'.
            self.act = nn.ReLU()
        
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=(1, 5),
                bias=False
            ),
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(2, 1),
                bias=False
            ),
            nn.BatchNorm2d(25),
            self.act,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=dropout)
        )
        
        layers = [25, 50, 100, 200]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=layers[i],
                    out_channels=layers[i + 1],
                    kernel_size=(1, 5),
                    bias=False
                ),
                nn.BatchNorm2d(layers[i + 1]),
                self.act,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=dropout))
            for i in range(len(layers) - 1)
        ])
        
        convs_output_size = 373
        flatten_size = 200 * reduce(lambda x, _: round((x - 4) / 2), layers[: -1], convs_output_size) # 8600.
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flatten_size, out_features=2, bias=True)
        )
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.first_conv(x) # [b, 25, 1, 373].
        for conv in self.convs: # [b, 200, 1, 43].
            x = conv(x)
        output = self.linear(x) # [b, 2].
        
        return output
