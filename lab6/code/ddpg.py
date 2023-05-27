import argparse
from datetime import datetime
import gym
import itertools
import os
from pathlib import Path
import numpy as np
import random
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Adam

from atari_wrappers import deque


class GaussianNoise(object):
    def __init__(self, dim, mu=None, std=None) -> None:
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)

class ReplayMemory(object):
    __slots__ = ['buffer']

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition: tuple):
        """
        Append (state, action, reward, next_state, done) to the buffer
        :param transition: (state, action, reward, next_state, done)
        :return: None
        """
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size: int, device: str):
        """
        Sample a batch of transition tensors
        :param batch_size: batch size
        :param device: training device
        :return: a batch of transition tensors
        """
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))

class ActorNet(nn.Module):
    def __init__(self, 
            state_dim=8, action_dim=2, hidden_dim=(400, 300)) -> None:
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_features=state_dim,
                out_features=hidden_dim[0]),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim[0],
                out_features=hidden_dim[1]),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim[1],
                out_features=action_dim), 
            nn.Tanh()
        )
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)
    
class CriticNet(nn.Module):
    def __init__(self, 
            state_dim=8, action_dim=2, hidden_dim=(400, 300)) -> None:
        super().__init__()
        
        h1, h2 = hidden_dim
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, 
            x: torch.FloatTensor, action: torch.FloatTensor) -> torch.FloatTensor:
        x = self.critic_head(torch.cat([x, action], dim=1))
        return self.critic(x)

class Ddpg(object):
    def __init__(self, 
            batch_size: int, device: str, 
            capacity: int, lr_actor: float, lr_critic: float, 
            gamma: float, tau: float) -> None:
        self._actor_net = ActorNet().to(device)
        self._critic_net = CriticNet().to(device)
        
        self._target_actor_net = ActorNet().to(device)
        self._target_critic_net = CriticNet().to(device)
        
        # Init target network.
        self._target_actor_net.load_state_dict(self._actor_net.state_dict())
        self._target_critic_net.load_state_dict(self._critic_net.state_dict())

        self._actor_opt = Adam(self._actor_net.parameters(), lr=lr_actor)
        self._critic_opt = Adam(self._critic_net.parameters(), lr=lr_critic)

        self._action_noise = GaussianNoise(dim=2)
        self._memory = ReplayMemory(capacity=capacity)

        self.device = device
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        
    def select_action(self, state, noise=True):
        '''based on the behavior (actor) network and exploration noise'''
        with torch.no_grad():
            if noise:
                re = self._actor_net(torch.from_numpy(state).view(1, -1).to(self.device)) + \
                    torch.from_numpy(self._action_noise.sample()).view(1, -1).to(self.device)
            else:
                re = self._actor_net(torch.from_numpy(state).view(1, -1).to(self.device))
                
        return re.to('cpu').numpy().squeeze()
    
    def append(self, state, action, reward, next_state, done):
        self._memory.append(
            state, action, [reward / 100], next_state, [int(done)])
        
    def update(self):
        # update the behavior networks' actor & critic
        self._update_behavior_network(self.gamma)
        # update the target networks' actor & critic
        self._update_target_network(
            self._target_actor_net, self._actor_net, self.tau)
        self._update_target_network(
            self._target_critic_net, self._critic_net, self.tau)
        
    def _update_behavior_network(self, gamma):
        # Sample transitions batch.
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        # Update critic.
        q_value = self._critic_net(state, action)
        with torch.no_grad():
           a_next = self._target_actor_net(next_state)
           q_next = self._target_critic_net(next_state, a_next)
           q_target = reward + gamma * q_next * (1 - done)
           
        criterion = nn.MSELoss()
        critic_loss = criterion(q_value, q_target)

        self._actor_net.zero_grad()
        self._critic_net.zero_grad()
        critic_loss.backward()
        self._critic_opt.step()

        # Update actor.
        action = self._actor_net(state)
        actor_loss = -self._critic_net(state, action).mean()

        self._actor_net.zero_grad()
        self._critic_net.zero_grad()
        actor_loss.backward()
        self._actor_opt.step()
        
    @staticmethod
    def _update_target_network(target_net, net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            target.data.copy_((1 - tau) * target.data + tau * behavior.data)

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save({
                'actor': self._actor_net.state_dict(),
                'critic': self._critic_net.state_dict(),
                'target_actor': self._target_actor_net.state_dict(),
                'target_critic': self._target_critic_net.state_dict(),
                'actor_opt': self._actor_opt.state_dict(),
                'critic_opt': self._critic_opt.state_dict(),
            }, model_path)
        else:
            torch.save({
                'actor': self._actor_net.state_dict(),
                'critic': self._critic_net.state_dict(),
            }, model_path)
            
        print(f'Finish saving checkpoint to {model_path} ...')
    
    def load(self, model_path, checkpoint=False):
        try: 
            model = torch.load(model_path)
            self._actor_net.load_state_dict(model['actor'])
            self._critic_net.load_state_dict(model['critic'])
            if checkpoint:
                self._target_actor_net.load_state_dict(model['target_actor'])
                self._target_critic_net.load_state_dict(model['target_critic'])
                self._actor_opt.load_state_dict(model['actor_opt'])
                self._critic_opt.load_state_dict(model['critic_opt'])
                
            print(f'Finish loading checkpoint from {model_path} ...')
        except Exception as msg: 
            print(f'Failed to loading checkpoint from {model_path} ...\nMessage: \n{msg}')

class Solver(object): 
    def __init__(self, config: dict) -> None:
        self.env = gym.make('LunarLanderContinuous-v2')
        
        device = torch.device(f'cuda:{config.device[0]}' if torch.cuda.is_available() else 'cpu')
        print(f'Currently using device {device} ...')
        
        self.agent = Ddpg(
            config.batch_size, device, 
            config.capacity, config.lr_actor, config.lr_critic, 
            config.gamma, config.tau)
        
        self.timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        
        self.mkdirs(config)
        
        ckpt = Path(config.resume_ckpt.strip())
        if ckpt.is_file():
            self.agent.load(ckpt, checkpoint=True)
        
        self.num_train_episodes = config.num_train_episodes
        self.num_steps_warmup = config.num_steps_warmup
        self.seed = config.seed
            
    def train(self) -> None:
        print('Start training ...')
        total_steps = 0
        ewma_reward = 0
        for episode in range(self.num_train_episodes):
            total_reward = 0
            state = self.env.reset()
            for t in itertools.count(start=1):
                # select action
                if total_steps < self.num_steps_warmup:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.select_action(state)
                # execute action
                next_state, reward, done, _ = self.env.step(action)

                # store transition
                self.agent.append(state, action, reward, next_state, done)
                if total_steps >= self.num_steps_warmup:
                    self.agent.update()

                state = next_state
                total_reward += reward
                total_steps += 1
                if done:
                    ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                    self.writer.add_scalar('Train/Episode Reward', total_reward, total_steps)
                    self.writer.add_scalar('Train/Ewma Reward', ewma_reward, total_steps)
                    print(f'Step: {total_steps}\tEpisode: {episode}\tLength: {t:3d}\tTotal reward: {total_reward:.2f}\tEwma reward: {ewma_reward:.2f}')
                    break
        self.env.close()
        
        self.agent.save(Path(self.dir_model, 'ddpg.pth'), checkpoint=True)
        
    def eval(self) -> None:
        print('Start testing ...')
        seeds = (self.seed + i for i in range(10))
        rewards = []
        for n_episode, seed in enumerate(seeds):
            total_reward = 0
            self.env.seed(seed)
            state = self.env.reset()
            for t in itertools.count(start=1):
                self.env.render()
                # select action
                action = self.agent.select_action(state, noise=False)
                # execute action
                next_state, reward, done, _ = self.env.step(action)

                state = next_state
                total_reward += reward

                if done:
                    self.writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                    print(f'Total reward: {total_reward:.2f}')
                    rewards.append(total_reward)
                    break

        print(f'Average Reward: {np.mean(rewards):.2f}')
        self.env.close()
        
    def mkdirs(self, config: dict) -> None:
        self.path = (
            f'{self.timestamp}_ddpg-lunar-lander-continuous-v2_'
            f'batch_size{config.batch_size}_episodes{config.num_train_episodes}warmup{config.num_steps_warmup}_'
            f'lra{config.lr_actor}lrc{config.lr_critic}gamma{config.gamma}tau{config.tau}'
        )
            
        self.writer = SummaryWriter(
            log_dir=Path(config.dir_writer, self.path))
        
        if config.mode == 'train':   
            self.dir_model = Path(config.dir_model, self.path)
            
            dir_model = Path(self.dir_model)
            if not dir_model.is_dir(): 
                dir_model.mkdir(parents=True, exist_ok=True)
                
def main(is_logging=True) -> None:
    # CUDA debugging.
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], 
        help='Mode. Options: "train" (default) or "test".')
    parser.add_argument('--device', type=lambda s: [int(item) for item in s.split(',')], default=[0],  
        help='GPUs.')

    """
    Data configs.
    """
    parser.add_argument('--batch_size', type=int, default=64, 
        help='Mini-batch size.')
    parser.add_argument('--num_workers', type=int, default=8, 
        help='Number of subprocesses to use for data loading.')
    
    """
    Model configs.
    """
    parser.add_argument('--capacity', type=int, default=500000,  
        help='Memory capacity.')
    
    """
    Training configs.
    """
    parser.add_argument('--num_train_episodes', type=int, default=1200, 
        help='Number of episodes during training.')
    parser.add_argument('--num_steps_warmup', type=int, default=10000,  
        help='Number of warmup steps.')
    parser.add_argument('--lr_actor', type=float, default=1e-3, 
        help='Learning rate of actor (DDPG).')
    parser.add_argument('--lr_critic', type=float, default=1e-3, 
        help='Learning rate of critic (DDPG).')
    parser.add_argument('--gamma', type=float, default=.99, 
        help='Gamma.')
    parser.add_argument('--tau', type=float, default=5e-3, 
        help='Tau (DDPG).')
    parser.add_argument('--seed', type=int, default=20200519, 
        help='Random seed.')
    
    """
    Testing configs.
    """
    parser.add_argument('--resume_ckpt', type=str, default='', 
        help='Checkpoint to load (only for evaluation).')
    
    """
    Paths.
    """
    parser.add_argument('--dir_writer', type=str, default='./runs/', 
        help='Directory of SummaryWriter.')
    parser.add_argument('--dir_model', type=str, default='./models/', 
        help='Directory of model weights.')
    
    config = parser.parse_args()
    if is_logging: 
        print(f'{config}\n')

    solver = Solver(config)
    if config.mode == 'train':
        solver.train()
    else: # 'test'.
        solver.eval()

if __name__ == '__main__':
    """
    Note: if test on server, 
        - Install dependencies: 
            - apt install freeglut3-dev
            - apt-get install xvfb
        - Run code below: 
            - xvfb-run -s "-screen 0 1400x900x24" python ddpg.py --mode test [args...]
            - e.g.
                - xvfb-run -s "-screen 0 1400x900x24" python ddpg.py --mode test --device 1 --resume_ckpt models/230521_063715_ddpg-lunar-lander-continuous-v2_batch_size64_episodes1200warmup10000_lra0.001lrc0.001gamma0.99tau0.005/ddpg.pth
    """
    
    main(True)