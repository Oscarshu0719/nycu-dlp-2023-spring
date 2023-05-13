import torch
import torch.nn as nn
from torch.autograd import Variable


class Lstm(nn.Module):
    def __init__(self, 
            input_size: int, output_size: int, hidden_size: int, num_layers: int, 
            batch_size: int, device: str) -> None:
        super().__init__()
        
        self.device = device
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([
            nn.LSTMCell(hidden_size, hidden_size) 
                for _ in range(self.num_layers)])
        self.output = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.Tanh())
        
        self.hidden = self.init_hidden()

    def init_hidden(self) -> list:
        hidden = []
        for _ in range(self.num_layers):
            hidden.append((
                Variable(torch.zeros(self.batch_size, self.hidden_size).to(self.device)),
                Variable(torch.zeros(self.batch_size, self.hidden_size).to(self.device))))
            
        return hidden

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        embedded = self.embed(input)
        h_in = embedded
        for i in range(self.num_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)

class GaussianLstm(nn.Module):
    def __init__(self, 
            input_size: int, output_size: int, hidden_size: int, num_layers: int, 
            batch_size: int, device: str) -> None:
        super().__init__()
        
        self.device = device
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([
            nn.LSTMCell(hidden_size, hidden_size) 
                for _ in range(self.num_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        
        self.hidden = self.init_hidden()

    def init_hidden(self) -> list:
        hidden = []
        for _ in range(self.num_layers):
            hidden.append((
                Variable(torch.zeros(self.batch_size, self.hidden_size).to(self.device)),
                Variable(torch.zeros(self.batch_size, self.hidden_size).to(self.device))))
            
        return hidden

    def reparameterize(self, 
            mu: torch.FloatTensor, logvar: torch.FloatTensor) -> torch.FloatTensor:
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        
        return mu + eps * std
        
    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        embedded = self.embed(input)
        h_in = embedded
        for i in range(self.num_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
            
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
