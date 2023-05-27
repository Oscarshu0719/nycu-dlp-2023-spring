import argparse
from datetime import datetime
import itertools
import os
from pathlib import Path
import numpy as np
import random
from tensorboardX import SummaryWriter
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from atari_wrappers import make_atari, wrap_deepmind


class ReplayMemory(object):
    def __init__(self, 
            capacity: int, state_shape: tuple, device: str) -> None:
        c, h, w = state_shape
        self.capacity = capacity
        self.device = device
        
        self.states = torch.zeros((capacity, c, h, w), dtype=torch.uint8)
        self.actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool)
        
        self.position = 0
        self.size = 0

    def push(self, state, action, reward, done):
        """Saves a transition."""
        self.states[self.position] = torch.from_numpy(state.squeeze(-1))
        self.actions[self.position, 0] = action
        self.rewards[self.position, 0] = reward
        self.dones[self.position, 0] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = max(self.size, self.position)

    def sample(self, batch_size: int):
        """Sample a batch of transitions"""
        i = torch.randint(0, high=self.size, size=(batch_size, ))
        
        state = self.states[i, : 4].to(self.device)
        action = self.actions[i].to(self.device)
        reward = self.rewards[i].to(self.device).float()
        next_state = self.states[i, 1: ].to(self.device)
        done = self.dones[i].to(self.device).float()
        
        return state, action, reward, next_state, done

    def __len__(self):
        return self.size

class Net(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes)
        )
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
                
class Dqn(object):
    def __init__(self, 
            batch_size: int, device: str, 
            state_shape: tuple, num_actions: int, 
            capacity: int, lr: float, gamma: float, 
            freq_update_behavior: int, freq_update_target: int) -> None:
        self._behavior_net = Net().to(device)
        self._target_net = Net().to(device)
        
        # Init target network.
        self._target_net.load_state_dict(
            self._behavior_net.state_dict())
        self._target_net.eval()

        self._optim = Adam(self._behavior_net.parameters(), lr=lr, eps=1.5e-4)
        self._memory = ReplayMemory(capacity, state_shape, device)

        ## config ##
        self.device = device
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.freq_update_behavior = freq_update_behavior
        self.freq_update_target = freq_update_target

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.from_numpy(state).unsqueeze(0).squeeze(-1)
                state = state.repeat(4, 1, 1).unsqueeze(0).to(self.device)
                action = self._behavior_net(state).max(1)[1].to('cpu').view(1, 1)
                return action.numpy()[0, 0].item()
        else: 
            return action_space.sample()
        
    def append(self, state, action, reward, done):
        """Push a transition into replay buffer"""
        self._memory.push(
            state, action, reward, int(done))
        
    def update(self, total_steps):
        if total_steps % self.freq_update_behavior == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.freq_update_target == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size)
        
        q_value = self._behavior_net(state).gather(dim=1, index=action.long())
        with torch.no_grad():
            q_next = self._target_net(next_state).max(dim=1)[0]
            q_target = reward[:, 0] + gamma * q_next * (1 - done[:, 0])
            
        loss = F.smooth_l1_loss(q_value, q_target.unsqueeze(1))
        
        self._optim.zero_grad()
        loss.backward()
        for param in self._behavior_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optim.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
                'target_net': self._target_net.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])
                
class Solver(object): 
    def __init__(self, config: dict) -> None:
        device = torch.device(f'cuda:{config.device[0]}' if torch.cuda.is_available() else 'cpu')
        print(f'Currently using device {device} ...')
        
        env = self.get_env()
        
        _, h, w = self.fp(env.reset()).shape
        self.agent = Dqn(
            config.batch_size, device, 
            (5, h, w), env.action_space.n, 
            config.capacity, config.lr, config.gamma, 
            config.freq_update_behavior, config.freq_update_target)
        
        env.close()
        
        self.timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        
        self.mkdirs(config)
        
        ckpt = Path(config.resume_ckpt.strip())
        if ckpt.is_file():
            self.agent.load(ckpt, checkpoint=True)
            
        self.num_train_episodes = config.num_train_episodes
        self.num_steps_warmup = config.num_steps_warmup
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        self.freq_eval = config.freq_eval
        self.epsilon_eval = config.epsilon_eval
        self.num_eval_episodes = config.num_eval_episodes
        
    def get_env(self): 
        env_raw = make_atari('BreakoutNoFrameskip-v4')
        return wrap_deepmind(env_raw)
            
    def train(self) -> None:
        print('Start training ...')
        env = self.get_env()
        
        action_space = env.action_space
        total_steps, epsilon = 0, 1.
        ewma_reward = 0
        for episode in range(self.num_train_episodes):
            total_reward = 0
            state = env.reset()
            state, reward, done, _ = env.step(1)
            for t in itertools.count(start=1): 
                # select action
                if total_steps < self.num_steps_warmup:
                    action = action_space.sample()
                else:
                    action = self.agent.select_action(state, epsilon, action_space)
                    epsilon -= (1 - self.epsilon_min) / self.epsilon_decay
                    epsilon = max(epsilon, self.epsilon_min)
                # execute action
                state, reward, done, _ = env.step(action)
                
                # store transition
                self.agent.append(state, action, reward, done)

                # update
                if total_steps >= self.num_steps_warmup:
                    self.agent.update(total_steps)

                total_reward += reward
                if (total_steps + 1) % self.freq_eval == 0:
                    self.eval()
                    self.agent.save(
                        Path(self.dir_model, f'dqn_breakout_{total_steps + 1}pt.pth'), checkpoint=False)
                
                total_steps += 1
                if done:
                    ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                    self.writer.add_scalar('Train/Episode Reward', total_reward, total_steps)
                    self.writer.add_scalar('Train/Ewma Reward', ewma_reward, total_steps)
                    print(f'Step: {total_steps}\tEpisode: {episode}\tLength: {t:3d}\tTotal reward: {total_reward:.2f}\tEwma reward: {ewma_reward:.2f}\tEpsilon: {epsilon:.3f}')
                    break
        env.close()
        
        self.agent.save(Path(self.dir_model, 'dqn_breakout.pth'), checkpoint=True)
        
    def eval(self) -> None:
        print('Start testing ...')
        env = self.get_env()
        
        action_space = env.action_space
        rewards = []
        for n_episode in range(self.num_eval_episodes):
            e_reward = 0
            done = False
            state = env.reset()
            
            while not done:
                time.sleep(0.01)
                env.render()
                action = self.agent.select_action(state, self.epsilon_eval, action_space)
                state, reward, done, _ = env.step(action)
                e_reward += reward
            
            print(f'Episode {n_episode + 1}: {e_reward:.2f}')
            rewards.append(e_reward)
        env.close()
        print(f'Average Reward: {np.mean(rewards):.2f}')
        
    def mkdirs(self, config: dict) -> None:
        self.path = (
            f'{self.timestamp}_dqn-breakout_'
            f'batch_size{config.batch_size}_episodes{config.num_train_episodes}warmup{config.num_steps_warmup}_'
            f'lr{config.lr}gamma{config.gamma}decay{config.epsilon_decay}min{config.epsilon_min}_'
            f'freq-behavior{config.freq_update_behavior}freq-target{config.freq_update_target}freq-eval{config.freq_eval}_'
            f'eps-eval{config.epsilon_eval}episodes-eval{config.num_eval_episodes}'
        )
            
        self.writer = SummaryWriter(
            log_dir=Path(config.dir_writer, self.path))
        
        if config.mode == 'train': 
            self.dir_model = Path(config.dir_model, self.path)
                
            dir_model = Path(self.dir_model)
            if not dir_model.is_dir(): 
                dir_model.mkdir(parents=True, exist_ok=True)
                
    def fp(self, num_frames: torch.tensor) -> torch.tensor:
        num_frames = torch.from_numpy(num_frames)
        h = num_frames.shape[-2]
        return num_frames.view(1, h, h)
                
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
    parser.add_argument('--batch_size', type=int, default=32, 
        help='Mini-batch size.')
    parser.add_argument('--num_workers', type=int, default=8, 
        help='Number of subprocesses to use for data loading.')
    
    """
    Model configs.
    """
    parser.add_argument('--capacity', type=int, default=100000,  
        help='Memory capacity.')
    
    """
    Training configs.
    """
    parser.add_argument('--num_train_episodes', type=int, default=20000,  
        help='Number of episodes during training.')
    parser.add_argument('--num_steps_warmup', type=int, default=20000,  
        help='Number of warmup steps.')
    parser.add_argument('--lr', type=float, default=6.25e-5, 
        help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=.99, 
        help='Gamma.')
    parser.add_argument('--epsilon_decay', type=float, default=1000000, 
        help='Decay ratio of epsilon.')
    parser.add_argument('--epsilon_min', type=float, default=1e-1, 
        help='Min of epsilon.')
    parser.add_argument('--freq_update_behavior', type=int, default=4, 
        help='Frequency of behavior network update.')
    parser.add_argument('--freq_update_target', type=int, default=10000, 
        help='Frequency of target network update.')
    parser.add_argument('--freq_eval', type=int, default=200000, 
        help='Frequency of target network update during evaluation.')
    
    """
    Testing configs.
    """
    parser.add_argument('--resume_ckpt', type=str, default='', 
        help='Checkpoint to load (only for evaluation).')
    parser.add_argument('--epsilon_eval', type=float, default=1e-2, 
        help='Epsilon during evaluation.')
    parser.add_argument('--num_eval_episodes', type=int, default=10, 
        help='Number of episodes during evaluation.')
    
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
            - xvfb-run -s "-screen 0 1400x900x24" python dqn_breakout.py --mode test [args...]
            - e.g.
                - xvfb-run -s "-screen 0 1400x900x24" python dqn_breakout.py --mode test --device 1 --resume_ckpt models/230526_120821_dqn-breakout_batch_size32_episodes20000warmup20000_lr6.25e-05gamma0.99decay1000000min0.1_freq-behavior4freq-target10000freq-eval200000_eps-eval0.01episodes-eval10/dqn-breakout.pth
    """
    
    main(True)