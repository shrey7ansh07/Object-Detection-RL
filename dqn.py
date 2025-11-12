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
from visual_feature_extractor import VisualFeatureExtractor
from enviornment import DQNObjectDetectionEnv
from ddqnagent import DoubleDQNAgent
from components import DQNetwork, Transition, ReplayBuffer



VISUAL_INPUT_SIZE = (224, 224) 
VISUAL_EMBEDDING_DIM = 128 
VIT_BASE_DIM = 768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


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
        log_interval: int = 500
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
        self.log_fh = open(self.save_dir / 'training_log.txt', 'a', buffering=1,encoding='utf-8')

        self.log_file = self.save_dir / 'training_log.txt'

    def log(self, message):
        """Log to file and console."""
        # print(message)
        # with open(self.log_file, 'a',encoding='utf-8') as f:
        #     f.write(message + '\n')
        print(message)
        self.log_fh.write(message + '\n')

    def train(self, start_epoch: int, num_epochs: int, warmup_steps: int = 1000):
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
        
        for epoch in range(start_epoch,start_epoch+num_epochs):
            self.log(f"\n{'='*70}")
            self.log(f"EPOCH {epoch+1}/{num_epochs}")
            self.log(f"{'='*70}")
            
            epoch_rewards = []
            epoch_ious = []
            epoch_steps_list = []
            epoch_losses = []

            for i in tqdm(range(len(self.dataset)), desc=f"Epoch {epoch+1}",mininterval=2.0):
                image_np, gt_box = self.dataset[i]
                env = DQNObjectDetectionEnv(
                        image=image_np,
                        ground_truth_box=gt_box,
                        feature_extractor=self.agent.policy_net.feature_extractor,
                        max_steps=20,
                        move_percentage=0.1,
                        initial_box="center",
                        device=self.device
                    )
                for ep in range(self.episodes_per_epoch):
                    
                    
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
                            f"epsilon: {self.agent.epsilon_start:.4f}"
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
            # if (epoch + 1) % 5 == 0:
            #     self.plot_training_progress()

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
        self.log(f"Checkpoint saved: {checkpoint_path}")

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Double DQN for Object Detection')
    parser.add_argument('--image_dir', type=str, default="dataset/images/train", 
                        help='Directory containing images')
    parser.add_argument('--annotations', type=str, default="dataset/labels/train_resized.json", 
                        help='Path to annotations JSON')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_ddqn', 
                        help='Checkpoint directory')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=10, 
                        help='Number of epochs (increased from 5)')
    parser.add_argument('--episodes_per_epoch', type=int, default=20, 
                        help='Episodes per image (increased from 10)')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Batch size (increased from 64)')
    parser.add_argument('--warmup_steps', type=int, default=5000, 
                        help='Steps to fill replay buffer before training')
    
    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=5e-5, 
                        help='Learning rate (lowered from 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.95, 
                        help='Discount factor (lowered from 0.99)')
    parser.add_argument('--buffer_size', type=int, default=50000, 
                        help='Replay buffer size (increased from 10000)')
    parser.add_argument('--epsilon_start', type=float, default=1.0, 
                        help='Starting epsilon')
    parser.add_argument('--epsilon_end', type=float, default=0.01, 
                        help='Minimum epsilon (lowered from 0.05)')
    parser.add_argument('--epsilon_decay', type=float, default=5e-6, 
                        help='Epsilon decay rate (slower decay)')
    parser.add_argument('--target_update_freq', type=int, default=1000, 
                        help='Target network update frequency')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=50, 
                        help='Logging interval')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\n")
    
    # Constants
    GEOMETRIC_DIM = 5
    STATE_DIM = GEOMETRIC_DIM + VISUAL_EMBEDDING_DIM  # 133
    ACTION_SIZE = 9
    
    # Initialize feature extractor
    print(f"Initializing Frozen Vision Transformer (ViT) with {VISUAL_EMBEDDING_DIM}D output...")
    feature_extractor_instance = VisualFeatureExtractor(output_dim=VISUAL_EMBEDDING_DIM).to(device)
    print("Feature extractor loaded\n")
    
    # Initialize Double DQN Agent
    print("Initializing Double DQN Agent...")
    agent = DoubleDQNAgent(
        state_dim=STATE_DIM,
        action_size=ACTION_SIZE,
        device=device,
        lr=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update_freq,
        buffer_size=args.buffer_size
    )
    # Load dataset
    print("Loading dataset...")
    dataset = ObjectDetectionDataset(args.image_dir, args.annotations)
    print(f"Dataset loaded: {len(dataset)} images\n")
    # Initialize trainer
    trainer = DQNTrainer(
        agent=agent,
        dataset=dataset,
        save_dir=args.save_dir,
        episodes_per_epoch=args.episodes_per_epoch,
        batch_size=args.batch_size,
        device=device,
        log_interval=args.log_interval
    )
    # Inject feature extractor
    agent.policy_net.feature_extractor = feature_extractor_instance
    agent.target_net.feature_extractor = feature_extractor_instance
    print("Double DQN Agent initialized\n")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Filter out feature_extractor keys
        policy_state = {k: v for k, v in checkpoint['policy_net_state_dict'].items() 
                       if not k.startswith('feature_extractor.')}
        target_state = {k: v for k, v in checkpoint['target_net_state_dict'].items() 
                       if not k.startswith('feature_extractor.')}
        
        # Load with strict=False to ignore missing feature_extractor keys
        agent.policy_net.load_state_dict(policy_state, strict=False)
        agent.target_net.load_state_dict(target_state, strict=False)
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon_start = checkpoint['epsilon']
        
        # Restore training history if available
        if 'rewards' in checkpoint:
            trainer.episode_rewards = checkpoint['rewards']
        if 'ious' in checkpoint:
            trainer.episode_ious = checkpoint['ious']
        if 'steps' in checkpoint:
            trainer.episode_steps = checkpoint['steps']
        if 'losses' in checkpoint:
            agent.losses = checkpoint['losses']
        
        checkpoint_epoch = checkpoint.get('epoch', 0)
        # print('checkpoint_epoch:', checkpoint_epoch)
        if isinstance(checkpoint_epoch, str):
            # If it's a string (like 'final_interrupted'), try to extract number
            # or default to 0 if we can't
            import re
            match = re.search(r'\d+', checkpoint_epoch)
            start_epoch = int(match.group()) if match else 0
            print(f"Resumed from checkpoint: {checkpoint_epoch}")
        else:
            start_epoch = int(checkpoint_epoch)
            print(f"Resumed from epoch {start_epoch}")
        
        print(f"  Epsilon: {agent.epsilon_start:.4f}")
        print(f"  Previous episodes: {len(trainer.episode_rewards)}\n")
    
    print('This is agent.epsilon start: ',agent.epsilon_start)
    
    
    
    
    # Training
    try:
        trainer.train(start_epoch,num_epochs=int(args.num_epochs), warmup_steps=args.warmup_steps)
        trainer.log("\n" + "="*70)
        trainer.log("TRAINING COMPLETED SUCCESSFULLY!")
        trainer.log("="*70)
        trainer.log_fh.close()
        # DQNTrainer.log_fh.close()

    except KeyboardInterrupt:
        trainer.log("\n\nTraining interrupted by user.")
        trainer.log("Saving final checkpoint...")
        trainer.save_checkpoint('final_interrupted')
        trainer.log("Final checkpoint saved.")
    except Exception as e:
        trainer.log(f"\n\nError during training: {str(e)}")
        trainer.log("Saving emergency checkpoint...")
        trainer.save_checkpoint('emergency')
        raise
    finally:
        trainer.log_fh.close()


if __name__ == '__main__':
    main()