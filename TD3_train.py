"""
TD3 Training Main Script
Works with your modular structure

File structure expected:
- enviornment.py (your environment)
- visual_feature_extractor.py (ViT feature extractor)
- components.py (dataset, etc.)
- td3agent.py (the TD3 agent from previous artifact)
- dimension.py (constants)
"""

import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# Import from your modules
from enviornment import DQNObjectDetectionEnv, ContinuousActionWrapper  # Your environment
from visual_feature_extractor import VisualFeatureExtractor  # ViT extractor
from components import ObjectDetectionDataset  # Your dataset loader
from TD3_agent import TD3Agent  # Your TD3 agent
from components import VISUAL_EMBEDDING_DIM, GEOMETRIC_DIM   # Constants


class TD3Trainer:
    """Trainer for TD3 object detection"""
    
    def __init__(
        self,
        agent: TD3Agent,
        dataset: ObjectDetectionDataset,
        feature_extractor: VisualFeatureExtractor,
        save_dir: str,
        episodes_per_image: int = 20,
        batch_size: int = 512,
        device: torch.device = torch.device('cpu'),
        log_interval: int = 50,
        action_scale: float = 0.1
    ):
        self.agent = agent
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_per_image = episodes_per_image
        self.batch_size = batch_size
        self.device = device
        self.log_interval = log_interval
        self.action_scale = action_scale
        
        # Tracking
        self.episode_rewards = []
        self.episode_ious = []
        self.episode_steps = []
        
        # Logging
        self.log_file = self.save_dir / 'training_log.txt'
        self.log_fh = open(self.log_file, 'a', buffering=1)


        self.image_cache = {}  # filename -> numpy array

    
    def log(self, message):
        """Log to file and console"""
        print(message)
        try:
            self.log_fh.write(message + '\n')
        except Exception:
            # fallback to safe write
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')


    def preload_feature_maps(self, max_cache: int = None):
        """
        Precompute and cache ViT feature maps for all images (or subset).

        Args:
            max_cache: Optional limit on number of images to cache (for debugging / memory control)
        """
        self.feature_cache = {}
        n_images = len(self.dataset) if max_cache is None else min(max_cache, len(self.dataset))

        self.log(f"🔄 Precomputing ViT feature maps for {n_images} images...")

        for i in tqdm(range(n_images), desc="Caching ViT feature maps"):
            image_np, gt_box = self.dataset[i]
            img_pil = Image.fromarray(image_np)

            with torch.no_grad():
                feature_map, resize_info = self.feature_extractor.forward_feature_map(img_pil)

            # Keep on the same device as the extractor (usually GPU)
            feature_map = feature_map.squeeze(0).to(self.device)
            self.feature_cache[i] = {
                "feature_map": feature_map,
                "resize_info": resize_info,
                "gt_box": gt_box,
            }

        self.log(f"✅ Cached feature maps for {len(self.feature_cache)} images.")
    def train(self, num_epochs: int, warmup_steps: int = 2000):
        """
        Train TD3 agent
        
        Args:
            num_epochs: Number of epochs to train
            warmup_steps: Steps to fill replay buffer before training
        """
        self.log("\n" + "="*70)
        self.log("TD3 TRAINING FOR OBJECT DETECTION")
        self.log("="*70)
        self.log(f"Total epochs: {num_epochs}")
        self.log(f"Episodes per image: {self.episodes_per_image}")
        self.log(f"Dataset size: {len(self.dataset)}")
        self.log(f"Warmup steps: {warmup_steps}")
        self.log(f"Action scale: {self.action_scale}")
        self.log(f"Replay buffer size: {len(self.agent.memory.buffer)}")
        self.log("="*70 + "\n")
        
        total_steps = 0
        
        for epoch in range(num_epochs):
            self.log(f"\n{'='*70}")
            self.log(f"EPOCH {epoch+1}/{num_epochs}")
            self.log(f"{'='*70}")
            
            epoch_rewards = []
            epoch_ious = []
            epoch_steps_list = []
            epoch_actor_losses = []
            epoch_critic_losses = []
            
            for img_idx in tqdm(range(len(self.dataset)), desc=f"Epoch {epoch+1}"):
                if hasattr(self, "feature_cache") and img_idx in self.feature_cache:
                    cache_entry = self.feature_cache[img_idx]
                    feature_map = cache_entry["feature_map"]
                    resize_info = cache_entry["resize_info"]
                    gt_box = cache_entry["gt_box"]

                    # Initialize environment using cached features
                    base_env = DQNObjectDetectionEnv(
                        image=None,   # no need to pass the image anymore
                        ground_truth_box=gt_box,
                        feature_extractor=None,  # not needed for cached mode
                        max_steps=20,
                        move_percentage=0.1,
                        initial_box="center",
                        device=self.device,
                        cached_feature_map=feature_map,
                        resize_info=resize_info
                    )

                else:
                    # Fallback if not cached
                    image_np, gt_box = self.dataset[img_idx]
                    base_env = DQNObjectDetectionEnv(
                        image=image_np,
                        ground_truth_box=gt_box,
                        feature_extractor=self.feature_extractor,
                        max_steps=20,
                        move_percentage=0.1,
                        initial_box="center",
                        device=self.device
                    )
                image_np, gt_box = self.dataset[img_idx]

            # Create base environment
                # base_env = DQNObjectDetectionEnv(
                #     image=image_np,
                #     ground_truth_box=gt_box,
                #     feature_extractor=self.feature_extractor,
                #     max_steps=20,
                #     move_percentage=0.1,  # Not used in continuous
                #     initial_box="center",
                #     device=self.device
                # )
                
                for ep in range(self.episodes_per_image):
                    
                    
                    # Wrap with continuous action wrapper
                    env = ContinuousActionWrapper(base_env, action_scale=self.action_scale)
                    
                    state = env.reset()
                    total_reward = 0
                    done = False
                    
                    while not done:
                        # Get continuous action from TD3
                        action = self.agent.act(state, add_noise=True)
                        
                        # Take step
                        next_state, reward, done, info = env.step(action)
                        
                        # Store transition
                        self.agent.memory.push(state, action, next_state, reward, done)
                        
                        state = next_state
                        total_reward += reward
                        total_steps += 1
                        
                        # Train TD3 (after warmup)
                        if total_steps > warmup_steps:
                            losses = self.agent.train_step(self.batch_size)
                            
                            if losses and losses['critic_loss'] is not None:
                                epoch_critic_losses.append(losses['critic_loss'])
                            if losses and losses['actor_loss'] is not None:
                                epoch_actor_losses.append(losses['actor_loss'])
                    
                    epoch_rewards.append(total_reward)
                    epoch_ious.append(info['iou'])
                    epoch_steps_list.append(info['step'])
                    
                    # Periodic logging
                    if (img_idx * self.episodes_per_image + ep) % self.log_interval == 0:
                        self.log(
                            f"  Image {img_idx+1}/{len(self.dataset)} | "
                            f"Episode {ep+1} | "
                            f"Steps: {info['step']:2d} | "
                            f"Reward: {total_reward:+7.2f} | "
                            f"IoU: {info['iou']:.4f} | "
                            f"Noise: {self.agent.exploration_noise:.4f}"
                        )
            
            # Epoch summary
            avg_reward = np.mean(epoch_rewards)
            avg_iou = np.mean(epoch_ious)
            avg_steps = np.mean(epoch_steps_list)
            avg_critic_loss = np.mean(epoch_critic_losses) if epoch_critic_losses else 0
            avg_actor_loss = np.mean(epoch_actor_losses) if epoch_actor_losses else 0
            
            self.episode_rewards.extend(epoch_rewards)
            self.episode_ious.extend(epoch_ious)
            self.episode_steps.extend(epoch_steps_list)
            
            self.log(f"\n{'='*70}")
            self.log(f"EPOCH {epoch+1} SUMMARY")
            self.log(f"{'='*70}")
            self.log(f"Average Reward:       {avg_reward:+7.2f}")
            self.log(f"Average Final IoU:    {avg_iou:.4f}")
            self.log(f"Average Steps:        {avg_steps:.2f}")
            self.log(f"Average Critic Loss:  {avg_critic_loss:.4f}")
            self.log(f"Average Actor Loss:   {avg_actor_loss:.4f}")
            self.log(f"Exploration Noise:    {self.agent.exploration_noise:.4f}")
            self.log(f"Replay Buffer Size:   {len(self.agent.memory)}")
            self.log(f"Total Steps:          {total_steps}")
            self.log(f"{'='*70}\n")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch + 1)
            
            # Plot progress
            if (epoch + 1) % 5 == 0:
                self.plot_training_progress()
    
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint"""
        checkpoint_path = self.save_dir / f'td3_checkpoint_epoch_{epoch}.pth'
        
        # Save agent
        self.agent.save(checkpoint_path)
        
        # Save training history
        history_path = self.save_dir / f'training_history_epoch_{epoch}.npz'
        np.savez(
            history_path,
            rewards=np.array(self.episode_rewards),
            ious=np.array(self.episode_ious),
            steps=np.array(self.episode_steps)
        )
        
        self.log(f"Checkpoint saved: {checkpoint_path}")
    
    def plot_training_progress(self):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Raw')
        axes[0, 0].plot(self._smooth(self.episode_rewards, 50), linewidth=2, label='Smoothed')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # IoUs
        axes[0, 1].plot(self.episode_ious, alpha=0.3, label='Raw')
        axes[0, 1].plot(self._smooth(self.episode_ious, 50), linewidth=2, label='Smoothed')
        axes[0, 1].axhline(y=0.7, color='r', linestyle='--', label='Target IoU')
        axes[0, 1].set_title('Final IoU per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Steps
        axes[1, 0].plot(self.episode_steps, alpha=0.3, label='Raw')
        axes[1, 0].plot(self._smooth(self.episode_steps, 50), linewidth=2, label='Smoothed')
        axes[1, 0].set_title('Steps per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Losses
        if self.agent.critic_losses:
            axes[1, 1].plot(self.agent.critic_losses, alpha=0.3, label='Critic Loss')
            if self.agent.actor_losses:
                # Actor updates less frequently, so subsample for plotting
                actor_x = np.arange(0, len(self.agent.critic_losses), 
                                   len(self.agent.critic_losses) // len(self.agent.actor_losses))[:len(self.agent.actor_losses)]
                axes[1, 1].plot(actor_x, self.agent.actor_losses, alpha=0.3, label='Actor Loss')
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.save_dir / 'training_progress.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        self.log(f"Training plot saved: {plot_path}")
    
    @staticmethod
    def _smooth(data, window=50):
        """Smooth data with moving average"""
        if len(data) < window:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2)
            smoothed.append(np.mean(data[start:end]))
        return smoothed


def main():
    parser = argparse.ArgumentParser(description='Train TD3 for Object Detection')
    
    # Data
    parser.add_argument('--image_dir', type=str, default="dataset/images/train")
    parser.add_argument('--annotations', type=str, default="dataset/labels/train_resized.json")
    parser.add_argument('--save_dir', type=str, default='./checkpoints_td3')
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--episodes_per_image', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    
    # TD3 Hyperparameters
    parser.add_argument('--lr_actor', type=float, default=3e-4)
    parser.add_argument('--lr_critic', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--action_scale', type=float, default=0.1, 
                       help='Scale for continuous actions (0.1 = 10% max change)')
    
    # Misc
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # State dimension
    STATE_DIM = GEOMETRIC_DIM + VISUAL_EMBEDDING_DIM  # 5 + 128 = 133
    
    # Initialize feature extractor
    print(f"Initializing feature extractor...")
    feature_extractor = VisualFeatureExtractor(output_dim=VISUAL_EMBEDDING_DIM).to(device)
    print("Feature extractor loaded\n")
    
    # Initialize TD3 agent
    print("Initializing TD3 Agent...")
    agent = TD3Agent(
        state_dim=STATE_DIM,
        action_dim=4,  # [dx, dy, dw, dh]
        device=device,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        tau=args.tau,
        buffer_size=args.buffer_size
    )
    print("TD3 Agent initialized\n")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        agent.load(args.resume)
        print(f"Resumed training\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = ObjectDetectionDataset(args.image_dir, args.annotations)
    print(f"Dataset loaded: {len(dataset)} images\n")
    
    # Initialize trainer
    trainer = TD3Trainer(
        agent=agent,
        dataset=dataset,
        feature_extractor=feature_extractor,
        save_dir=args.save_dir,
        episodes_per_image=args.episodes_per_image,
        batch_size=args.batch_size,
        device=device,
        log_interval=args.log_interval,
        action_scale=args.action_scale
    )
    # Precompute feature maps (optional but strongly recommended)
    trainer.preload_feature_maps(max_cache=500)  # adjust depending on GPU memory

    # Train
    try:
        trainer.train(num_epochs=args.num_epochs, warmup_steps=args.warmup_steps)
        trainer.log("\n" + "="*70)
        trainer.log("TRAINING COMPLETED SUCCESSFULLY!")
        trainer.log("="*70)
        trainer.log_fh.close()
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