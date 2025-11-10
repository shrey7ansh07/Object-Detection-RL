
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models 
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm
import torchvision.transforms as T # NEW IMPORT for standard ViT transforms
from visual_feature_extractor import VisualFeatureExtractor
from components import DQNetwork, Transition, ReplayBuffer


class DQNAgent:
    def __init__(self, state_dim, action_size, device, lr, gamma, epsilon_start, epsilon_end, epsilon_decay, target_update_freq, buffer_size):
        self.state_dim = state_dim
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_count = 0

        self.policy_net = DQNetwork(state_dim, action_size).to(device)
        self.target_net = DQNetwork(state_dim, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(1).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def train_step(self, batch_size):
        if len(self.memory) < batch_size:
            return None

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda d: d is False, batch.done)), device=self.device, dtype=torch.bool)
        
        state_batch = torch.stack([torch.from_numpy(s) for s in batch.state]).float().to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).long().unsqueeze(-1)
        reward_batch = torch.tensor(batch.reward, device=self.device).float()
        
        non_final_next_states = torch.stack([torch.from_numpy(s) for s, is_final in zip(batch.next_state, batch.done) if is_final is False]).float().to(self.device)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(-1)

        # Compute V(s_{t+1}) = max_a Q_target(s_{t+1}, a) for non-final states
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        # Compute the expected Q values
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
