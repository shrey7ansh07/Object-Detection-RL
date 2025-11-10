
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
from enviornment import DQNObjectDetectionEnv
from components import DQNetwork, Transition, ReplayBuffer
VISUAL_INPUT_SIZE = (224, 224) 
VISUAL_EMBEDDING_DIM = 128 
VIT_BASE_DIM = 768


# ============================================================================
# DOUBLE DQN AGENT
# ============================================================================

class DoubleDQNAgent:
    """
    Double DQN Agent - Key improvement over standard DQN
    Uses policy network to SELECT actions, target network to EVALUATE them
    This reduces Q-value overestimation
    """
    def __init__(
        self, 
        state_dim, 
        action_size, 
        device, 
        lr=5e-5,  # Lower learning rate
        gamma=0.95,  # Slightly lower discount
        epsilon_start=1.0, 
        epsilon_end=0.01,  # Lower minimum
        epsilon_decay=5e-6,  # Slower decay
        target_update_freq=1000,  # More frequent updates
        buffer_size=50000  # Larger buffer
    ):
        self.state_dim = state_dim
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.update_count = 0

        # Create networks
        self.policy_net = DQNetwork(state_dim, action_size).to(device)
        self.target_net = DQNetwork(state_dim, action_size).to(device)
        
        # Initialize weights
        self.policy_net.init_weights()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        
        # Tracking
        self.losses = []

    def act(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon_start:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(1).item()

    def update_epsilon(self):
        """Decay epsilon."""
        self.epsilon_start = max(self.epsilon_end, self.epsilon_start - self.epsilon_decay)

    def train_step(self, batch_size):
        """
        DOUBLE DQN TRAINING STEP
        Key difference from standard DQN:
        - Use policy_net to SELECT best action
        - Use target_net to EVALUATE that action
        """
        if len(self.memory) < batch_size:
            return None

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda d: d is False, batch.done)), 
            device=self.device, 
            dtype=torch.bool
        )
        
        state_batch = torch.stack([torch.from_numpy(s) for s in batch.state]).float().to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).long().unsqueeze(-1)
        reward_batch = torch.tensor(batch.reward, device=self.device).float()
        
        non_final_next_states = torch.stack([
            torch.from_numpy(s) for s, is_final in zip(batch.next_state, batch.done) 
            if is_final is False
        ]).float().to(self.device)
        
        # Current Q values: Q(s, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(-1)

        # DOUBLE DQN: Compute next state values
        next_state_values = torch.zeros(batch_size, device=self.device)
        
        if non_final_mask.sum() > 0:
            with torch.no_grad():
                # Use POLICY network to SELECT best actions
                best_actions = self.policy_net(non_final_next_states).argmax(1, keepdim=True)
                
                # Use TARGET network to EVALUATE those actions
                next_state_values[non_final_mask] = self.target_net(
                    non_final_next_states
                ).gather(1, best_actions).squeeze(-1)
        
        # Compute expected Q values: R + gamma * V(s')
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # Compute Huber loss (less sensitive to outliers than MSE)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"  [Target network updated at step {self.update_count}]")
        
        self.losses.append(loss.item())
        return loss.item()
