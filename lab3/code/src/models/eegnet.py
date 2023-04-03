import numpy as np
import torch
from torch import nn


class Eegnet(nn.Module): 
    def __init__(self, act_name='relu', dropout=0.25) -> None: 
        super().__init__()
        
        if act_name == 'elu':
            self.act = nn.ELU()
        elif act_name == 'leakyrelu':
            self.act = nn.LeakyReLU()
        else: # 'relu'.
            self.act = nn.ReLU()
        
        self.conv_2d = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1, 51),
                stride=(1, 1),
                padding=(0, 25),
                bias=False
            ),
            nn.BatchNorm2d(16)
        )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(2, 1),
                stride=(1, 1),
                groups=16,
                bias=False
            ),
            nn.BatchNorm2d(32),
            self.act,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=dropout)
        )
        
        self.separable_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 15),
                stride=(1, 1),
                padding=(0, 7),
                bias=False
            ),
            nn.BatchNorm2d(32),
            self.act,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=dropout)
        )
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv_2d(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        output = self.linear(x)
        
        return output