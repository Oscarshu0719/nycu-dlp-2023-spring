import torch
import torch.nn as nn
from typing import Union


class ConvBlock(nn.Module):
    def __init__(self, 
            in_channels: int, out_channels: int, kernel_size: int, 
            stride: int, padding: int,
            dilation=1, groups=1, bias=False, 
            activation=nn.ReLU(inplace=True)) -> None:
        super().__init__()
        
        self.layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding, 
                dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)]
        if activation:
            self.layers.append(activation)
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        out = self.layers(x)
        
        return out
    
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, 
            in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        
        self.conv1 = ConvBlock(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False)
        self.conv2 = ConvBlock(
            out_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=False, activation=None)
        self.relu1 = nn.ReLU(inplace=True)
        
        if stride >= 2 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, 
                    stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else: 
            self.shortcut = nn.Sequential()
            
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        out = self.conv1(x)
        out = self.conv2(out)
        
        out = out + self.shortcut(x)
        out = self.relu1(out)
        
        return out
    
class BottleNeck(nn.Module):
    expansion = 4
    
    def __init__(self, 
            in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        
        self.conv1 = ConvBlock(
            in_channels, out_channels, kernel_size=1, 
            stride=1, padding=0, bias=False)
        self.conv2 = ConvBlock(
            out_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False)
        self.conv3 = ConvBlock(
            out_channels, out_channels * 4, kernel_size=1, 
            stride=1, padding=0, bias=False, activation=None)
        self.relu1 = nn.ReLU(inplace=True)

        if stride >= 2 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * 4, kernel_size=1, 
                    stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        out = out + self.shortcut(x)
        out = self.relu1(out)
        
        return out
    
class ResNet(nn.Module):
    def __init__(self, 
            block: Union[BasicBlock, BottleNeck], groups: list, num_classes: int, 
            dim_hidden=128, init_weights=True) -> None:
        super().__init__()
        
        self.channels = 64
        self.block = block
        
        self.conv1_x = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=self.channels, 
                kernel_size=7, stride=2, padding=3, bias=False), 
            nn.BatchNorm2d(self.channels), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self.__make_layers(64, groups[0], 1)
        self.conv3_x = self.__make_layers(128, groups[1], 2)
        self.conv4_x = self.__make_layers(256, groups[2], 2)
        self.conv5_x = self.__make_layers(512, groups[3], 2)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(
                in_features=512 * self.block.expansion, 
                out_features=dim_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=dim_hidden, out_features=num_classes), 
        )
        
        if init_weights:
            self.__init_weights()
        
    def __init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __make_layers(self, 
            channels: int, blocks: int, stride: int) -> nn.modules:
        layers = [
            self.block(self.channels, channels, stride)
        ]
        self.channels = channels * self.block.expansion
        for _ in range(blocks - 1): 
            layers.append(self.block(self.channels, channels, stride=1))
        layers = nn.Sequential(*layers)
        
        return layers

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        out = self.conv1_x(x)
        
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        
        out = self.classifier(out)
        
        return out
    
def ResNet_18(num_classes=5):
    return ResNet(block=BasicBlock, groups=[2, 2, 2, 2], num_classes=num_classes)

def ResNet_50(num_classes=5):
    return ResNet(block=BottleNeck, groups=[3, 4, 6, 3], num_classes=num_classes)

__all__ = ['ResNet_18', 'ResNet_50']

if __name__ == '__main__': 
    from torchinfo import summary
    
    summary(ResNet_18(num_classes=5), input_size=(4, 3, 512, 512))
