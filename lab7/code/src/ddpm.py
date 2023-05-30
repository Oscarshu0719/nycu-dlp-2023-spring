import numpy as np
import torch
import torch.nn as nn


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

class ContextUnet(nn.Module):
    def __init__(self, in_channels: int, n_feat=128, n_classes=24):
        super().__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(
            nn.AvgPool2d(7), 
            nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 8, 8),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, 
            x: torch.FloatTensor, c: torch.FloatTensor, t: torch.FloatTensor, 
            context_mask: torch.FloatTensor) -> torch.FloatTensor:
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on
        
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        context_mask = (-1 * (1 - context_mask)) # flip 0 <-> 1
        c = c * context_mask
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, 'beta1 and beta2 must be in (0, 1)'

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        'alpha_t': alpha_t,  # \alpha_t
        'oneover_sqrta': oneover_sqrta,  # 1/\sqrt{\alpha_t}
        'sqrt_beta_t': sqrt_beta_t,  # \sqrt{\beta_t}
        'alphabar_t': alphabar_t,  # \bar{\alpha_t}
        'sqrtab': sqrtab,  # \sqrt{\bar{\alpha_t}}
        'sqrtmab': sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        'mab_over_sqrtmab': mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class DDPM(nn.Module):
    def __init__(self, 
            num_classes: int, device: str, 
            n_feat=128, n_T=400, betas=(1e-4, 0.02), drop_prob=0.1) -> None:
        super().__init__()
        
        self.nn_model = ContextUnet(
            in_channels=3, n_feat=n_feat, n_classes=num_classes).to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.num_classes = num_classes
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.
        
        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c) + self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))

    def sample(self, n_sample, size, device, guide_w=0.0) -> torch.FloatTensor:
        # follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0, self.num_classes).to(device) # context for us just cycles throught the labels
        c_i = c_i.repeat(int(n_sample / c_i.shape[0]))

        c_i = nn.functional.one_hot(c_i, num_classes=self.num_classes).type(torch.float)
        
        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)
        
        # double the batch
        c_i = c_i.repeat(2, 1).float()
        context_mask = context_mask.repeat(2, 1)
        context_mask[n_sample: ] = 1. # makes second half of batch context free

        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_T, 0, -1):
            print(f'Sampling timestep {i} ...', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[: n_sample]
            eps2 = eps[n_sample: ]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[: n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store
    
    def generate(self, c_i, n_sample, size, device, guide_w=0.0) -> torch.FloatTensor: 
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = c_i.type(torch.float).to(device)
        
        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)
        
        # double the batch
        c_i = c_i.repeat(2, 1).float()
        context_mask = context_mask.repeat(2, 1)
        context_mask[n_sample: ] = 1. # makes second half of batch context free

        print()
        for i in range(self.n_T, 0, -1):
            print(f'Sampling timestep {i} ...', end='\r')
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[: n_sample]
            eps2 = eps[n_sample: ]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[: n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
        
        return x_i
    