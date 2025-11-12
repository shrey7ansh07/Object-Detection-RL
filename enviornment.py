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

VISUAL_INPUT_SIZE = (224, 224) 
VISUAL_EMBEDDING_DIM = 128 
VIT_BASE_DIM = 768



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
        device: torch.device = torch.device('cpu'),
        cached_feature_map: torch.Tensor = None,
        resize_info: Dict = None,
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

        self.cached_feature_map = cached_feature_map
        self.resize_info = resize_info

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


        # Precompute ViT feature map once per image
        img_pil = Image.fromarray(self.image)
        # self.feature_map, self.resize_info = self.feature_extractor.forward_feature_map(img_pil)
        if self.cached_feature_map is not None:
            self.feature_map = self.cached_feature_map
        else:
            self.feature_map, self.resize_info = self.feature_extractor.forward_feature_map(img_pil)

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
            reward += 7.0
        elif current_iou > 0.6:
            reward += 4.0
        elif current_iou > 0.5:
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
    
    # def _get_visual_embedding(self) -> np.ndarray:
    #     x_min, y_min, x_max, y_max = self.current_box.astype(int)
    #     x_min = max(0, min(x_min, self.img_width))
    #     y_min = max(0, min(y_min, self.img_height))
    #     x_max = max(0, min(x_max, self.img_width))
    #     y_max = max(0, min(y_max, self.img_height))
    #     patch_np = self.image[y_min:y_max, x_min:x_max, :]
    
    #     if patch_np.size == 0:
    #         return np.zeros(VISUAL_EMBEDDING_DIM, dtype=np.float32)
    #     patch_pil = Image.fromarray(patch_np)
        
    #     patch_tensor = self.transform(patch_pil).unsqueeze(0).to(self.device)
        
    #     with torch.no_grad():
    #         embedding = self.feature_extractor(patch_tensor).cpu().numpy().flatten()
            
    #     return embedding
    def _get_visual_embedding(self) -> np.ndarray:
        """
        Get visual embedding for current box by pooling from precomputed ViT feature map.
        """
        if not hasattr(self, 'feature_map'):
            # Fallback if feature map not ready
            return np.zeros(VISUAL_EMBEDDING_DIM, dtype=np.float32)

        # feature_map: (1, 768, Hf, Wf) -> drop batch dim
        feature_map = self.feature_map[0]
        torch.cuda.synchronize()
        embedding = self.feature_extractor.extract_region_embedding(feature_map, self.current_box, self.resize_info)
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
    




# ============================================================================
# CONTINUOUS ACTION ENVIRONMENT WRAPPER
# ============================================================================

class ContinuousActionWrapper:
    """
    Wrapper to convert continuous TD3 actions to box adjustments
    
    Continuous actions: [dx, dy, dw, dh] in range [-1, 1]
    Maps to: relative changes in box position and size
    """
    
    def __init__(self, base_env, action_scale=0.1):
        """
        Args:
            base_env: Your existing environment
            action_scale: How much to scale actions (0.1 = 10% max change)
        """
        self.env = base_env
        self.action_scale = action_scale
    
    def reset(self):
        return self.env.reset()
    
    def step(self, continuous_action):
        """
        Convert continuous action to box changes
        
        Args:
            continuous_action: [dx, dy, dw, dh] each in [-1, 1]
        
        Returns:
            next_state, reward, done, info
        """
        # Get current box
        x_min, y_min, x_max, y_max = self.env.current_box
        width = x_max - x_min
        height = y_max - y_min
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Apply continuous actions (scaled)
        dx, dy, dw, dh = continuous_action * self.action_scale
        
        # Update box
        center_x += dx * width
        center_y += dy * height
        width *= (1 + dw)
        height *= (1 + dh)
        
        # Reconstruct box
        next_box = np.array([
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2
        ], dtype=np.float32)
        
        # Clip to image boundaries
        next_box = self.env._clip_box(next_box)
        
        # Calculate IoU
        current_iou = self.env._calculate_iou(next_box, self.env.ground_truth_box)
        
        # Calculate reward with area penalty
        reward = self._calculate_reward_with_area_penalty(
            current_iou, 
            next_box, 
            self.env.ground_truth_box,
            self.env.previous_iou
        )
        
        # Update environment state
        self.env.current_box = next_box
        self.env.previous_iou = current_iou
        self.env.step_count += 1
        
        # Check if done
        done = self._check_done(current_iou)
        
        # Get next state
        next_state = self.env._get_state()
        
        info = {'iou': current_iou, 'step': self.env.step_count}
        
        return next_state, reward, done, info
    
    def _calculate_reward_with_area_penalty(
        self, 
        current_iou, 
        pred_box, 
        gt_box, 
        previous_iou
    ):
        """
        Improved reward with penalty for excessive area outside GT
        
        This prevents the box from growing too large to maximize IoU
        """
        # Base reward: IoU improvement
        iou_delta = current_iou - previous_iou
        
        if iou_delta > 0.01:
            reward = 10.0 * iou_delta
        elif iou_delta < -0.01:
            reward = -10.0 * abs(iou_delta)
        else:
            reward = -0.5
        
        # Progressive IoU bonuses
        if current_iou > 0.8:
            reward += 10.0
        elif current_iou > 0.7:
            reward += 5.0
        elif current_iou > 0.6:
            reward += 2.0
        
        # AREA PENALTY: Penalize excessive area outside ground truth
        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        
        # Calculate area outside GT (predicted area - intersection)
        x_min_inter = max(pred_box[0], gt_box[0])
        y_min_inter = max(pred_box[1], gt_box[1])
        x_max_inter = min(pred_box[2], gt_box[2])
        y_max_inter = min(pred_box[3], gt_box[3])
        
        inter_width = max(0, x_max_inter - x_min_inter)
        inter_height = max(0, y_max_inter - y_min_inter)
        intersection = inter_width * inter_height
        
        outside_area = pred_area - intersection
        
        # Penalty proportional to wasted area
        if pred_area > 0:
            area_waste_ratio = outside_area / gt_area
            
            # Strong penalty if predicted box is much larger than GT
            if area_waste_ratio > 0.5:  # More than 50% waste
                reward -= 5.0 * area_waste_ratio
            elif area_waste_ratio > 0.3:  # 30-50% waste
                reward -= 2.0 * area_waste_ratio
            elif area_waste_ratio > 0.1:  # 10-30% waste
                reward -= 1.0 * area_waste_ratio
        
        # Bonus for compact, accurate boxes
        if current_iou > 0.7 and outside_area / gt_area < 0.2:
            reward += 3.0  # Bonus for tight, accurate box
        
        return reward
    
    def _check_done(self, current_iou):
        """Check if episode should terminate"""
        if self.env.step_count >= self.env.max_steps:
            return True
        if current_iou > 0.85:
            return True
        
        # Early stopping if stuck
        if hasattr(self.env, 'iou_history') and len(self.env.iou_history) >= 6:
            recent_ious = self.env.iou_history[-6:]
            if max(recent_ious) - min(recent_ious) < 0.01 and current_iou > 0.5:
                return True
        
        return False
    
    @property
    def img_height(self):
        return self.env.img_height
    
    @property
    def img_width(self):
        return self.env.img_width
    
    @property
    def current_box(self):
        return self.env.current_box
    
    @property
    def ground_truth_box(self):
        return self.env.ground_truth_box
