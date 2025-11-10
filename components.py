
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import torchvision.transforms as T # NEW IMPORT for standard ViT transforms




VISUAL_INPUT_SIZE = (224, 224) 
VISUAL_EMBEDDING_DIM = 128 
VIT_BASE_DIM = 768

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
