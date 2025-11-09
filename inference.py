# """
# DQN Inference & Visualization for Object Detection
# --------------------------------------------------
# Imports core components from the training module (dqn_train_vit128.py)
# to reuse the same environment, feature extractor, and model structure.
# """

# import os
# import torch
# import cv2
# import imageio
# import numpy as np
# from PIL import Image
# from pathlib import Path
# import json

# # Import from your training module
# from dqn import (
#     DQNetwork,
#     DQNAgent,
#     VisualFeatureExtractor,
#     DQNObjectDetectionEnv,
#     VISUAL_EMBEDDING_DIM
# )

# def visualize_agent_video(agent, env, save_path="outputs/agent_tracking.mp4", fps=2, device='cpu', make_gif=False):
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     frames = []

#     ACTION_NAMES = [
#         "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN",
#         "GROW_WIDTH", "SHRINK_WIDTH", "GROW_HEIGHT", "SHRINK_HEIGHT", "STOP"
#     ]

#     base_img = cv2.cvtColor(env.image.copy(), cv2.COLOR_RGB2BGR)
#     state = env.reset()
#     done = False
#     step_idx = 0

#     while not done:
#         step_idx += 1
#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
#         with torch.no_grad():
#             q_vals = agent(state_tensor)
#         action = torch.argmax(q_vals, dim=1).item()

#         next_state, _, done, info = env.step(action)
#         x1, y1, x2, y2 = env.current_box.astype(int)
#         gx1, gy1, gx2, gy2 = env.ground_truth_box.astype(int)

#         frame = base_img.copy()
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#         cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
#         cv2.putText(frame, f"Step: {step_idx} | {ACTION_NAMES[action]}", (10, 25),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
#         cv2.putText(frame, f"IoU: {info['iou']:.3f}", (10, 55),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
#         frames.append(frame)
#         state = next_state

#     h, w = frames[0].shape[:2]
#     out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#     for f in frames:
#         out.write(f)
#     out.release()
#     print(f"🎥 Video saved to: {save_path}")

#     if make_gif:
#         gif_path = save_path.replace(".mp4", ".gif")
#         imageio.mimsave(gif_path, [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=fps)
#         print(f"🌀 GIF saved to: {gif_path}")


# if __name__ == "__main__":
#     checkpoint_path = "checkpoints_vit128/dqn_checkpoint_epoch_1.pth"
#     image_path = "dataset/images/train/im00001.jpg"
#     annotations_path = "dataset/labels/train_resized.json"

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # --- Load feature extractor ---
#     # feature_extractor = VisualFeatureExtractor(output_dim=VISUAL_EMBEDDING_DIM).to(device)
#     # inference.py (snippet)

#     # 1) create feature extractor exactly as in training
#     feature_extractor = VisualFeatureExtractor(output_dim=VISUAL_EMBEDDING_DIM).to(device)

#     # 2) create the DQ network(s)
#     state_dim = 5 + VISUAL_EMBEDDING_DIM
#     action_size = 9
#     agent = DQNetwork(state_dim, action_size).to(device)

#     # 3) re-attach the feature extractor as it was during training
#     agent.feature_extractor = feature_extractor

#     # If you also load a target_net in the checkpoint, do same for it (if you create one)
#     # target = DQNetwork(state_dim, action_size).to(device)
#     # target.feature_extractor = feature_extractor

#     # 4) load weights
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     print("✅ Loaded checkpoint ", checkpoint["policy_net_state_dict"].keys())
#     # agent.load_state_dict(checkpoint["policy_net_state_dict"])
#     print('this is agent keys: ',agent.state_dict().keys())
#     # --- Load image and ground-truth box ---
#     with open(annotations_path, "r") as f:
#         ann = json.load(f)
#     img_info = ann["images"][0]
#     img_name = img_info["file_name"]
#     bbox = ann["annotations"][0]["bbox"]
#     gt_box = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
#     image = np.array(Image.open(Path("dataset/images/train") / img_name).convert("RGB"))

#     # --- Load agent ---
#     state_dim = 5 + VISUAL_EMBEDDING_DIM
#     action_size = 9
#     agent = DQNetwork(state_dim, action_size).to(device)

#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     agent.load_state_dict(checkpoint["policy_net_state_dict"])
#     print(f"✅ Loaded checkpoint from {checkpoint_path}")

#     # --- Create environment ---
#     env = DQNObjectDetectionEnv(image, gt_box, feature_extractor, device=device)

#     # --- Run and visualize ---
#     visualize_agent_video(agent, env, save_path="outputs/agent_inference.mp4", fps=2, device=device, make_gif=True)


"""
DQN Object Detection Inference Script
Loads a trained DQN model and performs object detection on input images.
Saves a video showing the agent's detection process step-by-step.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FFMpegWriter, PillowWriter
import argparse
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional
import cv2

# ============================================================================
# MODEL DEFINITIONS (Must match training code)
# ============================================================================

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


class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_size, hidden_size=256):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNObjectDetectionEnv:
    """Inference environment for object detection."""
    
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
    GROW_WIDTH = 4
    SHRINK_WIDTH = 5
    GROW_HEIGHT = 6
    SHRINK_HEIGHT = 7
    STOP = 8
    
    ACTION_NAMES = [
        "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN",
        "GROW_WIDTH", "SHRINK_WIDTH", "GROW_HEIGHT", "SHRINK_HEIGHT", "STOP"
    ]
    
    def __init__(
        self,
        image: np.ndarray,
        feature_extractor: VisualFeatureExtractor,
        max_steps: int = 20,
        move_percentage: float = 0.1,
        initial_box: str = "center",
        device: torch.device = torch.device('cpu')
    ):
        self.image = image
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
        """Reset environment to initial state."""
        self.step_count = 0
        
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
        
        self.previous_iou = 0.0
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, bool]:
        """Execute one step in the environment."""
        self.step_count += 1
        next_box = self._apply_action(action)
        self.current_box = next_box
        
        done = self._check_done(action)
        next_state = self._get_state()
        
        return next_state, done
    
    def _get_visual_embedding(self) -> np.ndarray:
        """Extract visual features from current bounding box region."""
        x_min, y_min, x_max, y_max = self.current_box.astype(int)
        
        # Clip to valid image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(self.img_width, x_max)
        y_max = min(self.img_height, y_max)
        
        patch_np = self.image[y_min:y_max, x_min:x_max, :]
    
        if patch_np.size == 0:
            return np.zeros(VISUAL_EMBEDDING_DIM, dtype=np.float32)
        
        patch_pil = Image.fromarray(patch_np)
        patch_tensor = self.transform(patch_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.feature_extractor(patch_tensor).cpu().numpy().flatten()
            
        return embedding

    def _get_state(self) -> np.ndarray:
        """Get current state representation."""
        normalized_box = self.current_box.copy()
        normalized_box[0] /= self.img_width
        normalized_box[1] /= self.img_height
        normalized_box[2] /= self.img_width
        normalized_box[3] /= self.img_height
        
        geometric_features = np.concatenate([normalized_box, [self.previous_iou]])
        visual_embedding = self._get_visual_embedding()
        state = np.concatenate([geometric_features, visual_embedding])
        
        return state.astype(np.float32)
    
    def _apply_action(self, action: int) -> np.ndarray:
        """Apply action to current bounding box."""
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
        """Clip box to image boundaries."""
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
    
    def _check_done(self, action: int) -> bool:
        """Check if episode should terminate."""
        if action == self.STOP:
            return True
        if self.step_count >= self.max_steps:
            return True
        return False


# ============================================================================
# INFERENCE CLASS
# ============================================================================

class DQNInference:
    """Inference engine for DQN object detection."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on (cuda/cpu)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model parameters
        self.state_dim = 5 + VISUAL_EMBEDDING_DIM  # 133
        self.action_size = 9
        
        # Load checkpoint
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load feature extractor
        print("Loading feature extractor...")
        self.feature_extractor = VisualFeatureExtractor(output_dim=VISUAL_EMBEDDING_DIM).to(self.device)
        
        # Load DQN policy network
        self.policy_net = DQNetwork(self.state_dim, self.action_size).to(self.device)
        
        # Filter out feature_extractor keys from the state dict
        policy_state_dict = checkpoint['policy_net_state_dict']
        filtered_state_dict = {k: v for k, v in policy_state_dict.items() 
                               if not k.startswith('feature_extractor.')}
        
        # Load only the DQN network weights (not the feature extractor)
        self.policy_net.load_state_dict(filtered_state_dict, strict=True)
        self.policy_net.eval()
        
        print(f"Model loaded successfully! (Epoch {checkpoint.get('epoch', 'unknown')})")
        print(f"Loaded {len(filtered_state_dict)} parameters for DQN network")
    
    def predict(
        self,
        image_path: str,
        max_steps: int = 20,
        initial_box: str = "center",
        save_video: bool = True,
        video_path: Optional[str] = None,
        show_confidence: bool = True
    ) -> Tuple[np.ndarray, list]:
        """
        Perform object detection on an image.
        
        Args:
            image_path: Path to input image
            max_steps: Maximum steps for detection
            initial_box: Initial box placement ("center" or "random")
            save_video: Whether to save detection video
            video_path: Output video path (auto-generated if None)
            show_confidence: Whether to show Q-values in video
            
        Returns:
            final_box: Detected bounding box [x_min, y_min, x_max, y_max]
            history: List of (box, action, q_values) for each step
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        print(f"\nProcessing image: {image_path}")
        print(f"Image size: {image_np.shape}")
        
        # Create environment
        env = DQNObjectDetectionEnv(
            image=image_np,
            feature_extractor=self.feature_extractor,
            max_steps=max_steps,
            move_percentage=0.1,
            initial_box=initial_box,
            device=self.device
        )
        
        # Run detection
        state = env.reset()
        done = False
        history = []
        
        print("\nDetection process:")
        print("-" * 60)
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
                action = q_values.argmax()
            
            # Store history
            history.append({
                'step': env.step_count,
                'box': env.current_box.copy(),
                'action': action,
                'action_name': env.ACTION_NAMES[action],
                'q_values': q_values
            })
            
            print(f"Step {env.step_count:2d}: {env.ACTION_NAMES[action]:15s} | "
                  f"Box: [{env.current_box[0]:6.1f}, {env.current_box[1]:6.1f}, "
                  f"{env.current_box[2]:6.1f}, {env.current_box[3]:6.1f}] | "
                  f"Max Q: {q_values.max():6.3f}")
            
            # Take step
            state, done = env.step(action)
        
        final_box = env.current_box
        print("-" * 60)
        print(f"Detection complete in {env.step_count} steps")
        print(f"Final box: [{final_box[0]:.1f}, {final_box[1]:.1f}, {final_box[2]:.1f}, {final_box[3]:.1f}]")
        
        # Save video if requested
        if save_video:
            if video_path is None:
                video_path = Path(image_path).stem + "_detection.mp4"
            self._save_video(image_np, history, final_box, video_path, show_confidence)
        
        return final_box, history
    
    def _save_video(
        self,
        image: np.ndarray,
        history: list,
        final_box: np.ndarray,
        video_path: str,
        show_confidence: bool
    ):
        """Save detection process as video."""
        print(f"\nSaving video to {video_path}...")
        
        # Use OpenCV to create video (more reliable than matplotlib writers)
        frames = self._generate_frames(image, history, final_box, show_confidence)
        
        # Save with OpenCV
        if video_path.endswith('.mp4'):
            self._save_opencv_video(frames, video_path, fps=2)
        else:
            # Fallback to GIF
            video_path = video_path.replace('.mp4', '.gif') if video_path.endswith('.mp4') else video_path
            self._save_gif(frames, video_path, fps=2)
        
        print(f"Video saved successfully: {video_path}")
    
    def _generate_frames(
        self,
        image: np.ndarray,
        history: list,
        final_box: np.ndarray,
        show_confidence: bool
    ) -> list:
        """Generate all frames for the video."""
        frames = []
        fig, ax = plt.subplots(1, figsize=(10, 10))
        
        # Initial frame
        ax.clear()
        ax.imshow(image)
        ax.set_title("Initial State", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        init_box = history[0]['box']
        rect = patches.Rectangle(
            (init_box[0], init_box[1]),
            init_box[2] - init_box[0],
            init_box[3] - init_box[1],
            linewidth=3, edgecolor='yellow', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Convert to numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        # Frames for each step
        for i, step_data in enumerate(history):
            ax.clear()
            ax.imshow(image)
            
            box = step_data['box']
            action_name = step_data['action_name']
            q_values = step_data['q_values']
            
            title = f"Step {i+1}/{len(history)}: {action_name}"
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            color = 'red' if action_name == 'STOP' else 'cyan'
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=3, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            if show_confidence:
                q_text = f"Max Q-value: {q_values.max():.3f}"
                ax.text(
                    10, 30, q_text,
                    color='white', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
                )
            
            ax.axis('off')
            
            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(frame)
        
        # Final frame
        ax.clear()
        ax.imshow(image)
        ax.set_title("Final Detection", fontsize=14, fontweight='bold')
        
        rect = patches.Rectangle(
            (final_box[0], final_box[1]),
            final_box[2] - final_box[0],
            final_box[3] - final_box[1],
            linewidth=4, edgecolor='lime', facecolor='none',
            label='Detected Object'
        )
        ax.add_patch(rect)
        ax.legend(loc='upper right', fontsize=12)
        ax.axis('off')
        
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        
        plt.close(fig)
        return frames
    
    def _save_opencv_video(self, frames: list, video_path: str, fps: int = 2):
        """Save frames as MP4 video using OpenCV."""
        try:
            height, width = frames[0].shape[:2]
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
        except Exception as e:
            print(f"Failed to save MP4 with OpenCV: {e}")
            print("Falling back to GIF...")
            video_path = video_path.replace('.mp4', '.gif')
            self._save_gif(frames, video_path, fps)
    
    def _save_gif(self, frames: list, gif_path: str, fps: int = 2):
        """Save frames as GIF using PIL."""
        from PIL import Image
        
        # Convert frames to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in frames]
        
        # Save as GIF
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000/fps),
            loop=0
        )
        print(f"Saved as GIF: {gif_path}")
    
    def _save_video_old(
        self,
        image: np.ndarray,
        history: list,
        final_box: np.ndarray,
        video_path: str,
        show_confidence: bool
    ):
        """Save detection process as video (old matplotlib method)."""
        print(f"\nSaving video to {video_path}...")
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        
        # Try to use FFMpeg, fallback to GIF
        try:
            writer = FFMpegWriter(fps=2, metadata={'artist': 'DQN'}, bitrate=1800)
        except:
            print("FFMpeg not available, saving as GIF instead...")
            video_path = video_path.replace('.mp4', '.gif')
            writer = PillowWriter(fps=2)
        
        with writer.saving(fig, video_path, dpi=100):
            pass  # Old method placeholder - replaced by new implementation
    
    def visualize_result(
        self,
        image_path: str,
        detected_box: np.ndarray,
        ground_truth_box: Optional[np.ndarray] = None,
        save_path: Optional[str] = None
    ):
        """
        Visualize detection result.
        
        Args:
            image_path: Path to input image
            detected_box: Detected bounding box
            ground_truth_box: Optional ground truth box for comparison
            save_path: Path to save visualization
        """
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image_np)
        
        # Draw detected box
        det_rect = patches.Rectangle(
            (detected_box[0], detected_box[1]),
            detected_box[2] - detected_box[0],
            detected_box[3] - detected_box[1],
            linewidth=3, edgecolor='red', facecolor='none',
            label='Detected'
        )
        ax.add_patch(det_rect)
        
        # Draw ground truth if provided
        if ground_truth_box is not None:
            gt_rect = patches.Rectangle(
                (ground_truth_box[0], ground_truth_box[1]),
                ground_truth_box[2] - ground_truth_box[0],
                ground_truth_box[3] - ground_truth_box[1],
                linewidth=3, edgecolor='green', facecolor='none',
                label='Ground Truth'
            )
            ax.add_patch(gt_rect)
            
            # Calculate and display IoU
            iou = self._calculate_iou(detected_box, ground_truth_box)
            ax.text(
                10, 30, f'IoU: {iou:.3f}',
                color='white', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
            )
        
        ax.legend(loc='upper right', fontsize=12)
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
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
# MAIN SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='DQN Object Detection Inference')
    parser.add_argument('--checkpoint', type=str,default='checkpoints_vit128/dqn_checkpoint_epoch_3.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str,default='/home/guest/GB_DATASET/GBCU_1255/Group2/Object-Detection-RL/dataset/images/train/im00008.jpg',
                        help='Path to input image')
    parser.add_argument('--max_steps', type=int, default=15,
                        help='Maximum detection steps')
    parser.add_argument('--initial_box', type=str, default='center',
                        choices=['center', 'random'],
                        help='Initial box placement')
    parser.add_argument('--video_path', type=str, default="output_detection.mp4",
                        help='Output video path (default: auto-generated)')
    parser.add_argument('--no_video', action='store_true',
                        help='Disable video saving')
    parser.add_argument('--ground_truth', type=str, default="458.9103690685413,138.2811950790861,686.5659050966608,326.3444639718805",
                        help='Ground truth box as "x_min,y_min,x_max,y_max"')
    parser.add_argument('--save_viz', type=str, default="result_viz.jpg",
                        help='Path to save result visualization')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = DQNInference(
        checkpoint_path=args.checkpoint,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Parse ground truth if provided
    gt_box = None
    if args.ground_truth:
        try:
            gt_box = np.array([float(x) for x in args.ground_truth.split(',')])
            if len(gt_box) != 4:
                raise ValueError
        except:
            print("Warning: Invalid ground truth format. Expected 'x_min,y_min,x_max,y_max'")
            gt_box = None
    
    # Run detection
    detected_box, history = inference.predict(
        image_path=args.image,
        max_steps=args.max_steps,
        initial_box=args.initial_box,
        save_video=not args.no_video,
        video_path=args.video_path
    )
    
    # Visualize result
    if args.save_viz or gt_box is not None:
        save_path = args.save_viz if args.save_viz else args.image.replace('.jpg', '_result.jpg').replace('.png', '_result.png')
        inference.visualize_result(
            image_path=args.image,
            detected_box=detected_box,
            ground_truth_box=gt_box,
            save_path=save_path
        )
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(f"Detected box: [{detected_box[0]:.1f}, {detected_box[1]:.1f}, {detected_box[2]:.1f}, {detected_box[3]:.1f}]")
    if gt_box is not None:
        iou = inference._calculate_iou(detected_box, gt_box)
        print(f"IoU with ground truth: {iou:.4f}")


if __name__ == '__main__':
    # Example usage if running without arguments (for testing)
    import sys
    # if len(sys.argv) == 1:
    #     print("Example usage:")
    #     print("python inference.py --checkpoint checkpoints_vit128/dqn_checkpoint_epoch_5.pth --image test_image.jpg")
    #     print("\nWith ground truth and video:")
    #     print("python inference.py --checkpoint checkpoints_vit128/dqn_checkpoint_epoch_5.pth --image test_image.jpg --ground_truth '100,100,300,300' --save_viz result.jpg")
    #     print("\nWithout video:")
    #     print("python inference.py --checkpoint checkpoints_vit128/dqn_checkpoint_epoch_5.pth --image test_image.jpg --no_video")
    
    main()