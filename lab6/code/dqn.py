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

class Net(nn.Module):
    def __init__(self, 
        state_dim=8, action_dim=4, hidden_dim=(400, 300)) -> None:
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(in_features=state_dim,
                out_features=hidden_dim[0]),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim[0],
                out_features=hidden_dim[1]),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim[1],
                out_features=action_dim)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.layers(x)

class Dqn(object):
    def __init__(self, 
            batch_size: int, device: str, 
            capacity: int, lr: float, gamma: float, 
            freq_update_behavior: int, freq_update_target: int) -> None:
        self._behavior_net = Net().to(device)
        self._target_net = Net().to(device)
        
        # Init target network.
        self._target_net.load_state_dict(
            self._behavior_net.state_dict())
        
        self._optim = Adam(
            self._behavior_net.parameters(), lr=lr)
        self._memory = ReplayMemory(capacity=capacity)
        
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.freq_update_behavior = freq_update_behavior
        self.freq_update_target = freq_update_target
        
    def select_action(self, state, epsilon, action_space):
        """
        epsilon-greedy based on behavior network
        :param state: current state
        :param epsilon: probability
        :param action_space: action space of current game
        :return: an action
        """
        if random.random() > epsilon:
            with torch.no_grad():
                # Max element is in 2nd column (dim=1).
                state = torch.from_numpy(state).view(1, -1).to(self.device)
                return self._behavior_net(state).max(dim=1)[1].item()
        else:
            return action_space.sample()
        
    def append(self, state, action, reward, next_state, done):
        """
        Append a step to the memory
        :param state: current state
        :param action: best action
        :param reward: reward
        :param next_state: next state
        :param done: whether the game is finished
        :return: None
        """
        self._memory.append(
            state, [action], [reward / 10], next_state, [int(done)])
        
    def update(self, total_steps):
        """
        Update behavior networks and target networks
        :return: None
        """
        if total_steps % self.freq_update_behavior == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.freq_update_target == 0:
            self._update_target_network()
            
    def _update_behavior_network(self, gamma):
        """
        Update behavior network
        :param gamma: gamma
        :return: None
        """
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        q_value = self._behavior_net(state).gather(dim=1, index=action.long())
        with torch.no_grad():
            q_next = self._target_net(next_state).max(dim=1)[0].view(-1, 1)
            q_target = reward + gamma * q_next * (1 - done)
            
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)

        self._optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optim.step()
        
    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
                'target_net': self._target_net.state_dict(),
                'optimizer': self._optim.state_dict(),
            }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)
            
        print(f'Finish saving checkpoint to {model_path} ...')

    def load(self, model_path, checkpoint=False):
        try: 
            model = torch.load(model_path)
            self._behavior_net.load_state_dict(model['behavior_net'])
            if checkpoint:
                self._target_net.load_state_dict(model['target_net'])
                self._optim.load_state_dict(model['optimizer'])
            
            print(f'Finish loading checkpoint from {model_path} ...')
        except Exception as msg: 
            print(f'Failed to loading checkpoint from {model_path} ...\nMessage: \n{msg}')

class Solver(object): 
    def __init__(self, config: dict) -> None:
        self.env = gym.make('LunarLander-v2')
        
        device = torch.device(f'cuda:{config.device[0]}' if torch.cuda.is_available() else 'cpu')
        print(f'Currently using device {device} ...')
        
        self.agent = Dqn(
            config.batch_size, device, 
            config.capacity, config.lr, config.gamma, 
            config.freq_update_behavior, config.freq_update_target)
        
        self.timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
        
        self.mkdirs(config)
        
        ckpt = Path(config.resume_ckpt.strip())
        if ckpt.is_file():
            self.agent.load(ckpt, checkpoint=True)
        
        self.num_train_episodes = config.num_train_episodes
        self.num_steps_warmup = config.num_steps_warmup
        self.epsilon_decay = config.epsilon_decay
        self.epsilon_min = config.epsilon_min
        self.epsilon_eval = config.epsilon_eval
        self.seed = config.seed
        
    def train(self) -> None:
        print('Start training ...')
        action_space = self.env.action_space
        total_steps, epsilon = 0, 1.
        ewma_reward = 0
        for episode in range(self.num_train_episodes):
            total_reward = 0
            state = self.env.reset()
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            for t in itertools.count(start=1): 
                # select action
                if total_steps < self.num_steps_warmup:
                    action = action_space.sample()
                else:
                    action = self.agent.select_action(state, epsilon, action_space)
                # execute action
                next_state, reward, done, _ = self.env.step(action)

                # store transition
                self.agent.append(state, action, reward, next_state, done)

                # update
                if total_steps >= self.num_steps_warmup:
                    self.agent.update(total_steps)

                state = next_state
                total_reward += reward
                total_steps += 1
                if done:
                    ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                    self.writer.add_scalar('Train/Episode Reward', total_reward, total_steps)
                    self.writer.add_scalar('Train/Ewma Reward', ewma_reward, total_steps)
                    print(f'Step: {total_steps}\tEpisode: {episode}\tLength: {t:3d}\tTotal reward: {total_reward:.2f}\tEwma reward: {ewma_reward:.2f}\tEpsilon: {epsilon:.3f}')
                    break
        self.env.close()
        
        self.agent.save(Path(self.dir_model, 'dqn.pth'), checkpoint=True)
        
    def eval(self) -> None:
        print('Start testing ...')
        action_space = self.env.action_space
        epsilon = self.epsilon_eval
        seeds = (self.seed + i for i in range(10))
        rewards = []
        for n_episode, seed in enumerate(seeds):
            total_reward = 0
            self.env.seed(seed)
            state = self.env.reset()
            for t in itertools.count(start=1):  # play an episode
                self.env.render()
                # select action
                action = self.agent.select_action(state, epsilon, action_space)
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
            f'{self.timestamp}_dqn-lunar-lander-v2_'
            f'batch_size{config.batch_size}_episodes{config.num_train_episodes}warmup{config.num_steps_warmup}_'
            f'lr{config.lr}gamma{config.gamma}decay{config.epsilon_decay}min{config.epsilon_min}_'
            f'freq-behavior{config.freq_update_behavior}freq-target{config.freq_update_target}_'
            f'eps-eval{config.epsilon_eval}'
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
    parser.add_argument('--batch_size', type=int, default=128, 
        help='Mini-batch size.')
    parser.add_argument('--num_workers', type=int, default=8, 
        help='Number of subprocesses to use for data loading.')
    
    """
    Model configs.
    """
    parser.add_argument('--capacity', type=int, default=10000,  
        help='Memory capacity.')
    
    """
    Training configs.
    """
    parser.add_argument('--num_train_episodes', type=int, default=1200,
        help='Number of episodes during training.')
    parser.add_argument('--num_steps_warmup', type=int, default=10000,  
        help='Number of warmup steps.')
    parser.add_argument('--lr', type=float, default=5e-4, 
        help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=.99, 
        help='Gamma.')
    parser.add_argument('--epsilon_decay', type=float, default=.995, 
        help='Decay ratio of epsilon.')
    parser.add_argument('--epsilon_min', type=float, default=1e-2, 
        help='Min of epsilon.')
    parser.add_argument('--freq_update_behavior', type=int, default=4, 
        help='Frequency of behavior network update.')
    parser.add_argument('--freq_update_target', type=int, default=100,
        help='Frequency of target network update.')
    parser.add_argument('--seed', type=int, default=20200519, 
        help='Random seed.')
    
    """
    Testing configs.
    """
    parser.add_argument('--resume_ckpt', type=str, default='', 
        help='Checkpoint to load (only for evaluation).')
    parser.add_argument('--epsilon_eval', type=float, default=1e-3, 
        help='Epsilon during evaluation.')
    
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
            - xvfb-run -s "-screen 0 1400x900x24" python dqn.py --mode test [args...]
            - e.g.
                - xvfb-run -s "-screen 0 1400x900x24" python dqn.py --mode test --device 1 --resume_ckpt models/230520_085345_dqn-lunar-lander-v2_batch_size128_episodes1200warmup10000_lr0.0005gamma0.99decay0.995min0.01_freq-behavior4freq-target100_eps-eval0.001/dqn.pth
    """
    main(True)