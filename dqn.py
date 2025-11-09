"""
Complete DQN Training Pipeline for Object Detection with FROZEN VISION TRANSFORMER (ViT) Embeddings
Output Dimension: 128
"""

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


VISUAL_INPUT_SIZE = (224, 224) 
VISUAL_EMBEDDING_DIM = 128 
VIT_BASE_DIM = 768

class VisualFeatureExtractor(nn.Module):
    def __init__(self, output_dim=VISUAL_EMBEDDING_DIM):
        super(VisualFeatureExtractor, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.heads = nn.Identity()

        for param in self.vit.parameters():
            param.requires_grad = False
            
        self.projection_head = nn.Linear(VIT_BASE_DIM, output_dim)
        
        for param in self.projection_head.parameters():
            param.requires_grad = False
        
        self.vit.eval()
        self.projection_head.eval()

    def forward(self, x):
        with torch.no_grad():
             features = self.vit(x) 
        embedding = self.projection_head(features)
        return embedding

# ============================================================================
# PART 1: DQN ENVIRONMENT (UPDATED FOR ViT Preprocessing)
# ============================================================================

class DQNObjectDetectionEnv:
    """Reinforcement Learning environment for object localization using DQN."""
    
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
    GROW_WIDTH = 4
    SHRINK_WIDTH = 5
    GROW_HEIGHT = 6
    SHRINK_HEIGHT = 7
    STOP = 8
    
    def __init__(
        self,
        image: np.ndarray,
        ground_truth_box: Tuple[int, int, int, int],
        feature_extractor: VisualFeatureExtractor,
        max_steps: int = 20,
        move_percentage: float = 0.1,
        initial_box: str = "center",
        device: torch.device = torch.device('cpu')
    ):
        self.image = image
        self.ground_truth_box = np.array(ground_truth_box, dtype=np.float32)
        
        self.feature_extractor = feature_extractor.eval().to(device) 
        self.device = device
        self.max_steps = max_steps
        self.delta = move_percentage
        self.initial_box_mode = initial_box
        
        self.img_height, self.img_width = image.shape[:2]
        self.action_space_size = 9
        
        self.current_box = None
        self.step_count = 0
        self.previous_iou = 0.0
        
        # Define the standard image preprocessing transform for ViT
        self.transform = T.Compose([
            T.Resize(VISUAL_INPUT_SIZE, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def reset(self) -> np.ndarray:
        self.step_count = 0
        # Reset tracking variables
        self.previous_action = None
        self.action_history = []
        self.iou_history = []
        
        if self.initial_box_mode == "center":
            center_x = self.img_width / 2
            center_y = self.img_height / 2
            width = self.img_width * 0.5
            height = self.img_height * 0.5
            
            self.current_box = np.array([
                center_x - width / 2,
                center_y - height / 2,
                center_x + width / 2,
                center_y + height / 2
            ], dtype=np.float32)
        else:
            width = np.random.uniform(0.2, 0.8) * self.img_width
            height = np.random.uniform(0.2, 0.8) * self.img_height
            x_min = np.random.uniform(0, self.img_width - width)
            y_min = np.random.uniform(0, self.img_height - height)
            
            self.current_box = np.array([
                x_min, y_min, x_min + width, y_min + height
            ], dtype=np.float32)
        
        self.previous_iou = self._calculate_iou(self.current_box, self.ground_truth_box)
        self.iou_history.append(self.previous_iou)
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.step_count += 1
        next_box = self._apply_action(action)
        current_iou = self._calculate_iou(next_box, self.ground_truth_box)
        reward = self._calculate_reward(current_iou, action)
        
        self.current_box = next_box
        self.previous_iou = current_iou
        self.iou_history.append(current_iou)
        done = self._check_done(action, current_iou)
        next_state = self._get_state()
        
        info = {'iou': current_iou, 'step': self.step_count, 'action': action} 
        return next_state, reward, done, info
    
    def _calculate_reward(self, current_iou: float, action: int) -> float:
        """
        IMPROVED REWARD FUNCTION
        Key changes:
        - Scaled IoU improvements (not just ±1)
        - Stronger STOP incentives
        - Progressive bonuses for high IoU
        """
        iou_delta = current_iou - self.previous_iou
        
        # 1. IoU improvement reward (SCALED by magnitude)
        if iou_delta > 0.01:  # Significant improvement
            reward = 10.0 * iou_delta  # Scale by actual improvement
        elif iou_delta < -0.01:  # Significant worsening
            reward = -10.0 * abs(iou_delta)
        else:  # Small or no change
            reward = -0.5  # Small penalty for wasting time
        
        # 2. Progressive bonuses for absolute IoU thresholds
        if current_iou > 0.8:
            reward += 10.0
        elif current_iou > 0.7:
            reward += 5.0
        elif current_iou > 0.6:
            reward += 2.0
        
        # 3. STOP action rewards (CRITICAL for learning when to stop)
        if action == self.STOP:
            if current_iou > 0.75:
                reward += 25.0  # Big reward for stopping at excellent IoU
            elif current_iou > 0.65:
                reward += 15.0  # Good reward for stopping at good IoU
            elif current_iou > 0.5:
                reward += 5.0   # Small reward for stopping at ok IoU
            else:
                reward -= 15.0  # Penalty for stopping too early
        
        # 4. Step efficiency bonus (encourage faster convergence)
        if current_iou > 0.7 and self.step_count < 10:
            reward += 2.0  # Bonus for reaching good IoU quickly
        
        # Track action for next step
        self.previous_action = action
        if len(self.action_history) < 10:
            self.action_history.append(action)
        
        return reward
    
    def _check_done(self, action: int, current_iou: float) -> bool:
        """Improved termination conditions."""
        # Terminate if STOP action
        if action == self.STOP:
            return True
        
        # Terminate if max steps reached
        if self.step_count >= self.max_steps:
            return True
        
        # Terminate if very high IoU achieved
        if current_iou > 0.85:
            return True
        
        # Early stopping if stuck (no improvement in last 5 steps)
        if len(self.iou_history) >= 6:
            recent_ious = self.iou_history[-6:]
            if max(recent_ious) - min(recent_ious) < 0.01:  # Stuck
                if current_iou > 0.5:  # But at reasonable IoU
                    return True
        
        return False
    
    def _get_visual_embedding(self) -> np.ndarray:
        x_min, y_min, x_max, y_max = self.current_box.astype(int)
        x_min = max(0, min(x_min, self.img_width))
        y_min = max(0, min(y_min, self.img_height))
        x_max = max(0, min(x_max, self.img_width))
        y_max = max(0, min(y_max, self.img_height))
        patch_np = self.image[y_min:y_max, x_min:x_max, :]
    
        if patch_np.size == 0:
            return np.zeros(VISUAL_EMBEDDING_DIM, dtype=np.float32)
        patch_pil = Image.fromarray(patch_np)
        
        patch_tensor = self.transform(patch_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.feature_extractor(patch_tensor).cpu().numpy().flatten()
            
        return embedding

    def _get_state(self) -> np.ndarray:
        normalized_box = self.current_box.copy()
        normalized_box[0] /= self.img_width
        normalized_box[1] /= self.img_height
        normalized_box[2] /= self.img_width
        normalized_box[3] /= self.img_height
        geometric_features = np.concatenate([normalized_box, [self.previous_iou]]) # 5 dimensions
        
        # 2. Visual Features (ViT Embedding) - 128 dimensions
        visual_embedding = self._get_visual_embedding() 

        # 3. Concatenate
        state = np.concatenate([geometric_features, visual_embedding])
        return state.astype(np.float32)
    
    # ... (_apply_action, _clip_box, _calculate_reward, _check_done, _calculate_iou methods from the original code)
    
    def _apply_action(self, action: int) -> np.ndarray:
        # (This remains the same)
        x_min, y_min, x_max, y_max = self.current_box
        width = x_max - x_min
        height = y_max - y_min
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        dx = self.delta * width
        dy = self.delta * height
        
        if action == self.MOVE_LEFT:
            center_x -= dx
        elif action == self.MOVE_RIGHT:
            center_x += dx
        elif action == self.MOVE_UP:
            center_y -= dy
        elif action == self.MOVE_DOWN:
            center_y += dy
        elif action == self.GROW_WIDTH:
            width *= (1 + self.delta)
        elif action == self.SHRINK_WIDTH:
            width *= (1 - self.delta)
        elif action == self.GROW_HEIGHT:
            height *= (1 + self.delta)
        elif action == self.SHRINK_HEIGHT:
            height *= (1 - self.delta)
        
        next_box = np.array([
            center_x - width / 2, center_y - height / 2,
            center_x + width / 2, center_y + height / 2
        ], dtype=np.float32)
        
        return self._clip_box(next_box)
    
    def _clip_box(self, box: np.ndarray) -> np.ndarray:
        clipped = box.copy()
        clipped[0] = np.clip(clipped[0], 0, self.img_width)
        clipped[1] = np.clip(clipped[1], 0, self.img_height)
        clipped[2] = np.clip(clipped[2], 0, self.img_width)
        clipped[3] = np.clip(clipped[3], 0, self.img_height)
        
        if clipped[0] >= clipped[2]:
            clipped[2] = clipped[0] + 1
        if clipped[1] >= clipped[3]:
            clipped[3] = clipped[1] + 1
            
        return clipped
    
    # def _calculate_reward(self, current_iou: float, action: int) -> float:
    #     # (This remains the same)
    #     iou_delta = current_iou - self.previous_iou
        
    #     if iou_delta > 0:
    #         reward = 1.0
    #     elif iou_delta < 0:
    #         reward = -1.0
    #     else:
    #         reward = -0.1
        
    #     if current_iou > 0.7:
    #         reward += 5.0
        
    #     if action == self.STOP and current_iou > 0.7:
    #         reward += 3.0
        
    #     return reward
    
    # def _check_done(self, action: int, current_iou: float) -> bool:
    #     # (This remains the same)
    #     if action == self.STOP:
    #         return True
    #     if self.step_count >= self.max_steps:
    #         return True
    #     if current_iou > 0.95:
    #         return True
    #     return False

    @staticmethod
    def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        # (This remains the same)
        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])
        
        inter_width = max(0, x_max_inter - x_min_inter)
        inter_height = max(0, y_max_inter - y_min_inter)
        intersection = inter_width * inter_height
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# ============================================================================
# PART 2: DQN NETWORK (MODIFIED STATE DIMENSION)
# ============================================================================

class ReplayBuffer:
    # (ReplayBuffer class remains the same)
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_size, hidden_size=256):
        super(DQNetwork, self).__init__()
        # The input dimension (state_dim) is now 5 (geometric) + 128 (visual) = 133
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# class DQNAgent:
#     def __init__(self, state_dim, action_size, device, lr, gamma, epsilon_start, epsilon_end, epsilon_decay, target_update_freq, buffer_size):
#         self.state_dim = state_dim
#         self.action_size = action_size
#         self.device = device
#         self.gamma = gamma
#         self.epsilon = epsilon_start
#         self.epsilon_end = epsilon_end
#         self.epsilon_decay = epsilon_decay
#         self.target_update_freq = target_update_freq
#         self.update_count = 0

#         self.policy_net = DQNetwork(state_dim, action_size).to(device)
#         self.target_net = DQNetwork(state_dim, action_size).to(device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()
        
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
#         self.memory = ReplayBuffer(buffer_size)

#     def act(self, state):
#         if random.random() < self.epsilon:
#             return random.randrange(self.action_size)
#         else:
#             with torch.no_grad():
#                 state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
#                 q_values = self.policy_net(state_tensor)
#                 return q_values.argmax(1).item()

#     def update_epsilon(self):
#         self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

#     def train_step(self, batch_size):
#         if len(self.memory) < batch_size:
#             return None

#         transitions = self.memory.sample(batch_size)
#         batch = Transition(*zip(*transitions))

#         non_final_mask = torch.tensor(tuple(map(lambda d: d is False, batch.done)), device=self.device, dtype=torch.bool)
        
#         state_batch = torch.stack([torch.from_numpy(s) for s in batch.state]).float().to(self.device)
#         action_batch = torch.tensor(batch.action, device=self.device).long().unsqueeze(-1)
#         reward_batch = torch.tensor(batch.reward, device=self.device).float()
        
#         non_final_next_states = torch.stack([torch.from_numpy(s) for s, is_final in zip(batch.next_state, batch.done) if is_final is False]).float().to(self.device)
        
#         # Compute Q(s_t, a)
#         state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(-1)

#         # Compute V(s_{t+1}) = max_a Q_target(s_{t+1}, a) for non-final states
#         next_state_values = torch.zeros(batch_size, device=self.device)
#         with torch.no_grad():
#             next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
#         # Compute the expected Q values
#         expected_state_action_values = reward_batch + self.gamma * next_state_values

#         # Compute Huber loss
#         loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

#         # Optimize the model
#         self.optimizer.zero_grad()
#         loss.backward()
        
#         # Clip gradients
#         torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
#         self.optimizer.step()
        
#         self.update_count += 1
#         if self.update_count % self.target_update_freq == 0:
#             self.target_net.load_state_dict(self.policy_net.state_dict())

#         return loss.item()



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


class ObjectDetectionDataset:
    """
    Handles loading data formatted in the COCO style (list of images, list of annotations).
    """
    def __init__(self, image_dir: str, annotations_path: str):
        self.image_dir = Path(image_dir)
        
        with open(annotations_path, 'r') as f:
            full_data = json.load(f)

        # 1. Create a dictionary to map image_id to its metadata (width, height, file_name)
        image_metadata = {img['id']: img for img in full_data.get('images', [])}
        
        # 2. Create the final map linking file_name to its bounding box and size
        self.data_map: Dict[str, Any] = {}
        
        annotations_list = full_data.get('annotations', [])
        
        # COCO allows multiple annotations per image. We assume one for simplicity.
        # If there are multiple, this will only take the last one processed for a given image_id.
        for ann in annotations_list:
            image_id = ann['image_id']
            if image_id in image_metadata:
                metadata = image_metadata[image_id]
                file_name = metadata['file_name']
                
                # Check if the image file actually exists
                if (self.image_dir / file_name).exists():
                    self.data_map[file_name] = {
                        'bbox': ann['bbox'],  # [x, y, width, height] in COCO
                        'width': metadata['width'],
                        'height': metadata['height']
                    }
        
        self.image_file_names = list(self.data_map.keys())
        
        if not self.image_file_names:
             raise ValueError("No valid image files found after processing annotations.")

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        
        image_file_name = self.image_file_names[idx]
        data = self.data_map[image_file_name]
        
        # Load image
        image_path = self.image_dir / image_file_name
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        bbox_coco = data['bbox']
        
        x, y, x1, y1 = bbox_coco
        
        # Convert to [x_min, y_min, x_max, y_max] format (used by your DQN Env)
        ground_truth_box = (
            float(x), 
            float(y), 
            float(x1), 
            float(y1)
        )
        
        return image_np, ground_truth_box

# class DQNTrainer:
#     # (DQNTrainer class remains the same)
#     def __init__(self, agent: DQNAgent, dataset: ObjectDetectionDataset, save_dir: str, episodes_per_epoch: int, batch_size: int, device: torch.device):
#         self.agent = agent
#         self.dataset = dataset
#         self.save_dir = Path(save_dir)
#         self.save_dir.mkdir(parents=True, exist_ok=True)
#         self.episodes_per_epoch = episodes_per_epoch
#         self.batch_size = batch_size
#         self.device = device
        
#         self.episode_rewards = []
#         self.episode_ious = []

#     def load_checkpoint(self, checkpoint_path: str):
#         if not os.path.exists(checkpoint_path):
#             raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
#         print(f"Loading checkpoint from {checkpoint_path}...")
#         checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
#         self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
#         self.agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
#         self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.agent.epsilon = checkpoint.get('epsilon', 1.0)
        
#         # Continue previous logs if they exist
#         self.episode_rewards = checkpoint.get('rewards', [])
#         self.episode_ious = checkpoint.get('ious', [])
        
#         start_epoch = checkpoint.get('epoch', 0)
#         print(f"Resumed from epoch {start_epoch}")
#         return start_epoch
        
#     def train(self, num_epochs: int):
#         print("Starting DQN Training...")
        
#         for epoch in range(num_epochs):
#             print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
#             epoch_rewards = []
#             epoch_ious = []

#             # Iterate through each image in the dataset
#             for i in tqdm(range(len(self.dataset)), desc=f"Epoch {epoch+1}"):
#                 image_np, gt_box = self.dataset[i]
                
#                 # We run multiple episodes per image for better sample efficiency
#                 for _ in range(self.episodes_per_epoch):
#                     # NOTE: A new feature extractor is NOT created here. The SAME, 
#                     # frozen feature extractor instance is passed to all environments.
                    
#                     # Create a new environment for the episode
#                     env = DQNObjectDetectionEnv(
#                         image=image_np,
#                         ground_truth_box=gt_box,
#                         feature_extractor=self.agent.policy_net.feature_extractor,
#                         max_steps=15, 
#                         move_percentage=0.1, 
#                         initial_box="center",
#                         device=self.device
#                     )
                    
#                     state = env.reset()
#                     total_reward = 0
#                     done = False
                    
#                     while not done:
#                         action = self.agent.act(state)
#                         next_state, reward, done, info = env.step(action)
                        
#                         # Store the transition
#                         self.agent.memory.push(state, action, next_state, reward, done)
                        
#                         state = next_state
#                         total_reward += reward
                        
#                         loss = self.agent.train_step(self.batch_size)


#                     epoch_rewards.append(total_reward)
#                     epoch_ious.append(info['iou'])
#                     self.agent.update_epsilon()

#                 if (i + 1) % 100 == 0:
#                     avg_reward_100 = np.mean(epoch_rewards[-100:]) if len(epoch_rewards) >= 100 else np.mean(epoch_rewards)
#                     avg_iou_100 = np.mean(epoch_ious[-100:]) if len(epoch_ious) >= 100 else np.mean(epoch_ious)
#                     print(f"[Epoch {epoch+1}] Processed {i+1}/{len(self.dataset)} images | "
#                         f"Avg Reward (last 100): {avg_reward_100:.3f} | "
#                         f"Avg IoU (last 100): {avg_iou_100:.4f} | "
#                         f"Epsilon: {self.agent.epsilon:.4f}")

#             # Epoch summary and logging
#             avg_reward = np.mean(epoch_rewards)
#             avg_iou = np.mean(epoch_ious)
#             self.episode_rewards.extend(epoch_rewards)
#             self.episode_ious.extend(epoch_ious)

#             print(f"Epoch {epoch+1} finished.")
#             print(f"  Average Reward: {avg_reward:.2f}")
#             print(f"  Average Final IoU: {avg_iou:.4f}")
#             print(f"  Epsilon: {self.agent.epsilon:.4f}")

#             # Save checkpoint
#             self.save_checkpoint(epoch + 1)

#     def save_checkpoint(self, epoch: int):
#         # (save_checkpoint method remains the same)
#         checkpoint_path = self.save_dir / f'dqn_checkpoint_epoch_{epoch}.pth'
#         torch.save({
#             'epoch': epoch,
#             'policy_net_state_dict': self.agent.policy_net.state_dict(),
#             'target_net_state_dict': self.agent.target_net.state_dict(),
#             'optimizer_state_dict': self.agent.optimizer.state_dict(),
#             'epsilon': self.agent.epsilon,
#             'rewards': self.episode_rewards,
#             'ious': self.episode_ious
#         }, checkpoint_path)
#         print(f"Checkpoint saved to {checkpoint_path}")


# ============================================================================
# IMPROVED TRAINER WITH BETTER LOGGING
# ============================================================================


class DQNTrainer:
    def __init__(
        self, 
        agent: DoubleDQNAgent, 
        dataset: ObjectDetectionDataset, 
        save_dir: str, 
        episodes_per_epoch: int, 
        batch_size: int, 
        device: torch.device,
        log_interval: int = 100
    ):
        self.agent = agent
        self.dataset = dataset
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_per_epoch = episodes_per_epoch
        self.batch_size = batch_size
        self.device = device
        self.log_interval = log_interval
        
        self.episode_rewards = []
        self.episode_ious = []
        self.episode_steps = []
        
        # Logging
        self.log_file = self.save_dir / 'training_log.txt'

    def log(self, message):
        """Log to file and console."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def train(self, num_epochs: int, warmup_steps: int = 1000):
        """
        Train with warmup period.
        Warmup: Fill replay buffer before training starts.
        """
        self.log("\n" + "="*70)
        self.log("DOUBLE DQN TRAINING WITH IMPROVED REWARDS")
        self.log("="*70)
        self.log(f"Total epochs: {num_epochs}")
        self.log(f"Episodes per image: {self.episodes_per_epoch}")
        self.log(f"Dataset size: {len(self.dataset)}")
        self.log(f"Warmup steps: {warmup_steps}")
        self.log(f"Replay buffer size: {self.agent.memory.buffer.maxlen}")
        self.log(f"Learning rate: {self.agent.optimizer.param_groups[0]['lr']}")
        self.log(f"Gamma: {self.agent.gamma}")
        self.log(f"Epsilon: {self.agent.epsilon_start} -> {self.agent.epsilon_end}")
        self.log("="*70 + "\n")
        
        total_steps = 0
        
        for epoch in range(num_epochs):
            self.log(f"\n{'='*70}")
            self.log(f"EPOCH {epoch+1}/{num_epochs}")
            self.log(f"{'='*70}")
            
            epoch_rewards = []
            epoch_ious = []
            epoch_steps_list = []
            epoch_losses = []

            for i in tqdm(range(len(self.dataset)), desc=f"Epoch {epoch+1}"):
                image_np, gt_box = self.dataset[i]
                
                for ep in range(self.episodes_per_epoch):
                    env = DQNObjectDetectionEnv(
                        image=image_np,
                        ground_truth_box=gt_box,
                        feature_extractor=self.agent.policy_net.feature_extractor,
                        max_steps=20,
                        move_percentage=0.1,
                        initial_box="center",
                        device=self.device
                    )
                    
                    state = env.reset()
                    total_reward = 0
                    done = False
                    
                    while not done:
                        action = self.agent.act(state)
                        next_state, reward, done, info = env.step(action)
                        
                        self.agent.memory.push(state, action, next_state, reward, done)
                        
                        state = next_state
                        total_reward += reward
                        total_steps += 1
                        
                        # Train only after warmup
                        if total_steps > warmup_steps:
                            loss = self.agent.train_step(self.batch_size)
                            if loss is not None:
                                epoch_losses.append(loss)
                    
                    epoch_rewards.append(total_reward)
                    epoch_ious.append(info['iou'])
                    epoch_steps_list.append(info['step'])
                    
                    self.agent.update_epsilon()
                    
                    # Periodic logging
                    if (i * self.episodes_per_epoch + ep) % self.log_interval == 0:
                        self.log(
                            f"  Image {i+1}/{len(self.dataset)} | "
                            f"Episode {ep+1} | "
                            f"Steps: {info['step']:2d} | "
                            f"Reward: {total_reward:+7.2f} | "
                            f"IoU: {info['iou']:.4f} | "
                            f"ε: {self.agent.epsilon_start:.4f}"
                        )

            # Epoch summary
            avg_reward = np.mean(epoch_rewards)
            avg_iou = np.mean(epoch_ious)
            avg_steps = np.mean(epoch_steps_list)
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            
            self.episode_rewards.extend(epoch_rewards)
            self.episode_ious.extend(epoch_ious)
            self.episode_steps.extend(epoch_steps_list)

            self.log(f"\n{'='*70}")
            self.log(f"EPOCH {epoch+1} SUMMARY")
            self.log(f"{'='*70}")
            self.log(f"Average Reward:     {avg_reward:+7.2f}")
            self.log(f"Average Final IoU:  {avg_iou:.4f}")
            self.log(f"Average Steps:      {avg_steps:.2f}")
            self.log(f"Average Loss:       {avg_loss:.4f}")
            self.log(f"Current Epsilon:    {self.agent.epsilon_start:.4f}")
            self.log(f"Replay Buffer Size: {len(self.agent.memory)}")
            self.log(f"Total Steps:        {total_steps}")
            self.log(f"{'='*70}\n")

            # Save checkpoint
            self.save_checkpoint(epoch + 1)
            
            # Plot progress
            if (epoch + 1) % 5 == 0:
                self.plot_training_progress()

    def save_checkpoint(self, epoch: int):
        checkpoint_path = self.save_dir / f'ddqn_checkpoint_epoch_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'policy_net_state_dict': self.agent.policy_net.state_dict(),
            'target_net_state_dict': self.agent.target_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'epsilon': self.agent.epsilon_start,
            'rewards': self.episode_rewards,
            'ious': self.episode_ious,
            'steps': self.episode_steps,
            'losses': self.agent.losses
        }, checkpoint_path)
        self.log(f"✓ Checkpoint saved: {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description='Train DQN for Object Detection')
    parser.add_argument('--image_dir', type=str, default="dataset/images/train", help='Directory containing images')
    parser.add_argument('--annotations', type=str, default="dataset/labels/train_resized.json", help='Path to annotations JSON')
    parser.add_argument('--save_dir', type=str, default='./checkpoints2_vit128', help='Checkpoint directory')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--episodes_per_epoch', type=int, default=20, help='Episodes per image')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate (Often lower for DRL)')
    parser.add_argument('--buffer_size', type=int, default=50000, help='Replay buffer size')
    
    args = parser.parse_args()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    GEOMETRIC_DIM = 5
    STATE_DIM = GEOMETRIC_DIM + VISUAL_EMBEDDING_DIM  # 5 + 128 = 133
    ACTION_SIZE = 9
    
    print(f"Initializing Frozen Vision Transformer (ViT) with {VISUAL_EMBEDDING_DIM} output dimensions...")
    feature_extractor_instance = VisualFeatureExtractor(output_dim=VISUAL_EMBEDDING_DIM).to(device)
    
    # 2. Initialize DQN Agent
    agent = DoubleDQNAgent(
        state_dim=STATE_DIM,
        action_size=ACTION_SIZE,
        device=device,
        lr=args.learning_rate,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=5e-6, # Adjust based on expected number of total steps
        target_update_freq=1000,
        buffer_size=args.buffer_size
    )
    # Inject the feature extractor into the policy network for easy access in the environment
    agent.policy_net.feature_extractor = feature_extractor_instance 
    agent.target_net.feature_extractor = feature_extractor_instance # Keep consistent
    
    # 3. Load Data
    dataset = ObjectDetectionDataset(args.image_dir, args.annotations)
    print(f"Dataset loaded with {len(dataset)} images.")
    
    # 4. Initialize Trainer
    trainer = DQNTrainer(
        agent=agent,
        dataset=dataset,
        save_dir=args.save_dir,
        episodes_per_epoch=args.episodes_per_epoch,
        batch_size=args.batch_size,
        device=device
    )
    
    # 5. Start Training
    try:
        # resume_checkpoint = './checkpoints_vit128/dqn_checkpoint_epoch_1.pth'  # your ckpt path

        # start_epoch = 0
        # if os.path.exists(resume_checkpoint):
        #     start_epoch = trainer.load_checkpoint(resume_checkpoint)

        # Continue training for remaining epochs
        trainer.train(num_epochs=args.num_epochs)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final checkpoint...")
        trainer.save_checkpoint('final_interrupted')

if __name__ == '__main__':
    # NOTE: You may need to run this command in your terminal 
    # if you don't use an IDE that supports passing arguments:
    # python your_script_name.py --image_dir /path/to/images --annotations /path/to/annotations.json
    # The default paths are placeholders, please update them for your system.
    
    # Temporarily set placeholders if running in an environment without argparse
    if 'get_ipython' in globals():
        class MockArgs:
            image_dir = "dataset/images/train"
            annotations = "dataset/labels/train_resized.json"
            save_dir = './checkpoints_vit128'
            num_epochs = 2
            episodes_per_epoch = 2
            batch_size = 32
            learning_rate = 1e-4
            buffer_size = 10000
        
        main.__globals__['argparse'].ArgumentParser = lambda *args, **kwargs: MockArgs()
        main.__globals__['argparse'].ArgumentParser.add_argument = lambda *args, **kwargs: None

    main()