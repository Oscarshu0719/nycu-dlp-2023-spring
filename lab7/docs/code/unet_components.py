class ResidualConvBlock(nn.Module):
    def __init__(self, 
            in_channels: int, out_channels: int, is_res: bool=False) -> None:
        super().__init__()
        
        self.same_channels = (in_channels==out_channels)
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)

            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
                
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.layers = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels), 
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.FloatTensor, skip: torch.FloatTensor) -> torch.FloatTensor:
        x = torch.cat((x, skip), 1)
        x = self.layers(x)
        
        return x

class EmbedFC(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim))

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x.view(-1, self.input_dim)
        
        return self.layers(x)