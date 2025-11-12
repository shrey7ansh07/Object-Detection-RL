
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import torchvision.transforms as T # NEW IMPORT for standard ViT transforms
from PIL import Image
from typing import List, Dict, Tuple, Any
import json
from pathlib import Path
import torch


VISUAL_INPUT_SIZE = (224, 224) 
VISUAL_EMBEDDING_DIM = 128 
VIT_BASE_DIM = 768
GEOMETRIC_DIM = 5
STATE_DIM = GEOMETRIC_DIM + VISUAL_EMBEDDING_DIM  # 133

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

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
        self.image_cache = {}

        
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
        if image_file_name not in self.image_cache:
            image_path = self.image_dir / image_file_name
            image = Image.open(image_path).convert('RGB')
            self.image_cache[image_file_name] = np.array(image)
        image_np = self.image_cache[image_file_name]
        # Load image
        # image_path = self.image_dir / image_file_name
        # image = Image.open(image_path).convert('RGB')
        # image_np = np.array(image)
        
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

# ============================================================================
# PART 2: DQN NETWORK (MODIFIED STATE DIMENSION)
# ============================================================================

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



# ============================================================================
# TD3 NETWORKS
# ============================================================================

class Actor(nn.Module):
    """
    Actor Network: Maps state to continuous actions
    Output: 4 continuous values for [dx, dy, dw, dh] in range [-1, 1]
    """
    def __init__(self, state_dim, action_dim=4, hidden_size=256, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Last layer small weights for stable initial policy
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Tanh to bound actions to [-1, 1]
        action = self.max_action * torch.tanh(self.fc3(x))
        return action


class Critic(nn.Module):
    """
    Critic Network: Estimates Q(s, a)
    TD3 uses TWO critics to reduce overestimation
    """
    def __init__(self, state_dim, action_dim=4, hidden_size=256):
        super(Critic, self).__init__()
        
        # Q1 network
        self.fc1_1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc1_2 = nn.Linear(hidden_size, hidden_size)
        self.fc1_3 = nn.Linear(hidden_size, 1)
        
        # Q2 network
        self.fc2_1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2_2 = nn.Linear(hidden_size, hidden_size)
        self.fc2_3 = nn.Linear(hidden_size, 1)
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        
        # Q1 forward
        q1 = F.relu(self.fc1_1(sa))
        q1 = F.relu(self.fc1_2(q1))
        q1 = self.fc1_3(q1)
        
        # Q2 forward
        q2 = F.relu(self.fc2_1(sa))
        q2 = F.relu(self.fc2_2(q2))
        q2 = self.fc2_3(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """Return only Q1 value (for actor loss)"""
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.fc1_1(sa))
        q1 = F.relu(self.fc1_2(q1))
        q1 = self.fc1_3(q1)
        return q1

