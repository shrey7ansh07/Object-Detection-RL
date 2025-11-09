import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Dict, Any
from PIL import Image


class DQNObjectDetectionEnv:    
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
        max_steps: int = 15,
        move_percentage: float = 0.05,
        initial_box: str = "center"
    ):
        """
        Initialize the DQN Object Detection Environment.
        
        Args:
            image: Input image as NumPy array (H x W x 3)
            ground_truth_box: Ground truth bounding box (x_min, y_min, x_max, y_max)
            max_steps: Maximum number of steps per episode
            move_percentage: Percentage of box dimension to move (Delta = 0.1 means 10%)
            initial_box: How to initialize starting box ("center" or "random")
        """
        self.image = image
        self.ground_truth_box = np.array(ground_truth_box, dtype=np.float32)
        self.max_steps = max_steps
        self.delta = move_percentage
        self.initial_box_mode = initial_box
        
        # Image dimensions
        self.img_height, self.img_width = image.shape[:2]
        
        # Action space: 9 discrete actions
        self.action_space_size = 9
        
        # Episode tracking
        self.current_box = None
        self.step_count = 0
        self.previous_iou = 0.0
        
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state for a new episode.
        
        Returns:
            initial_state: The initial state representation
        """
        # Reset step counter
        self.step_count = 0
        
        # Initialize starting box
        if self.initial_box_mode == "center":
            # Start with a box in the center of the image (50% of image size)
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
        else:  # random
            # Random box with random position and size (20-80% of image)
            width = np.random.uniform(0.2, 0.8) * self.img_width
            height = np.random.uniform(0.2, 0.8) * self.img_height
            x_min = np.random.uniform(0, self.img_width - width)
            y_min = np.random.uniform(0, self.img_height - height)
            
            self.current_box = np.array([
                x_min, y_min, x_min + width, y_min + height
            ], dtype=np.float32)
        
        # Calculate initial IoU
        self.previous_iou = self._calculate_iou(self.current_box, self.ground_truth_box)
        
        # Return initial state
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action index (0-8)
            
        Returns:
            next_state: The next state after taking the action
            reward: The reward received
            done: Whether the episode has terminated
            info: Additional information (current IoU, etc.)
        """
        # Increment step counter
        self.step_count += 1
        
        # Apply action to get next box
        next_box = self._apply_action(action)
        
        # Calculate IoU for the new box
        current_iou = self._calculate_iou(next_box, self.ground_truth_box)
        
        # Calculate reward based on IoU change
        reward = self._calculate_reward(current_iou, action)
        
        # Update current box and IoU
        self.current_box = next_box
        self.previous_iou = current_iou
        
        # Check termination conditions
        done = self._check_done(action, current_iou)
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            'iou': current_iou,
            'step': self.step_count,
            'action': action
        }
        
        return next_state, reward, done, info
    
    def _apply_action(self, action: int) -> np.ndarray:
        """
        Apply the given action to the current box.
        
        Args:
            action: Action index (0-8)
            
        Returns:
            next_box: Updated bounding box coordinates
        """
        x_min, y_min, x_max, y_max = self.current_box
        
        # Calculate current box properties
        width = x_max - x_min
        height = y_max - y_min
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Calculate movement amounts
        dx = self.delta * width
        dy = self.delta * height
        
        # Apply action
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
        elif action == self.STOP:
            # No change for STOP action
            pass
        
        # Reconstruct box from center and dimensions
        next_box = np.array([
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2
        ], dtype=np.float32)
        
        # Clip to image boundaries
        next_box = self._clip_box(next_box)
        
        return next_box
    
    def _clip_box(self, box: np.ndarray) -> np.ndarray:
        """
        Clip bounding box coordinates to image boundaries.
        
        Args:
            box: Bounding box (x_min, y_min, x_max, y_max)
            
        Returns:
            clipped_box: Box clipped to valid image coordinates
        """
        clipped = box.copy()
        clipped[0] = np.clip(clipped[0], 0, self.img_width)   # x_min
        clipped[1] = np.clip(clipped[1], 0, self.img_height)  # y_min
        clipped[2] = np.clip(clipped[2], 0, self.img_width)   # x_max
        clipped[3] = np.clip(clipped[3], 0, self.img_height)  # y_max
        
        # Ensure min < max
        if clipped[0] >= clipped[2]:
            clipped[2] = clipped[0] + 1
        if clipped[1] >= clipped[3]:
            clipped[3] = clipped[1] + 1
            
        return clipped
    
    def _calculate_reward(self, current_iou: float, action: int) -> float:
        """
        Calculate reward based on IoU change.
        
        Args:
            current_iou: IoU after taking the action
            action: The action taken
            
        Returns:
            reward: Calculated reward value
        """
        # Reward based on IoU improvement
        iou_delta = current_iou - self.previous_iou
        
        # Scale reward by IoU change
        if iou_delta > 0:
            reward = 1.0  # Positive reward for improvement
        elif iou_delta < 0:
            reward = -1.0  # Negative reward for worsening
        else:
            reward = -0.1  # Small penalty for no change
        
        # Bonus for high IoU
        if current_iou > 0.7:
            reward += 5.0
        
        # Bonus for STOP action when IoU is good
        if action == self.STOP and current_iou > 0.7:
            reward += 3.0
        
        return reward
    
    def _check_done(self, action: int, current_iou: float) -> bool:
        """
        Check if episode should terminate.
        
        Args:
            action: The action taken
            current_iou: Current IoU value
            
        Returns:
            done: Whether episode is complete
        """
        # Terminate if STOP action
        if action == self.STOP:
            return True
        
        # Terminate if max steps reached
        if self.step_count >= self.max_steps:
            return True
        
        # Terminate if very high IoU achieved
        if current_iou > 0.85:
            return True
        
        return False
    
    def _get_state(self) -> np.ndarray:
        """
        Get the current state representation.
        
        Returns:
            state: State vector containing normalized box coordinates
        """
        # Normalize coordinates to [0, 1]
        normalized_box = self.current_box.copy()
        normalized_box[0] /= self.img_width   # x_min
        normalized_box[1] /= self.img_height  # y_min
        normalized_box[2] /= self.img_width   # x_max
        normalized_box[3] /= self.img_height  # y_max
        
        # State: [x_min, y_min, x_max, y_max, current_iou]
        state = np.concatenate([
            normalized_box,
            [self.previous_iou]
        ])
        
        return state.astype(np.float32)
    
    @staticmethod
    def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1: First bounding box (x_min, y_min, x_max, y_max)
            box2: Second bounding box (x_min, y_min, x_max, y_max)
            
        Returns:
            iou: IoU value between 0 and 1
        """
        # Calculate intersection coordinates
        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])
        
        # Calculate intersection area
        inter_width = max(0, x_max_inter - x_min_inter)
        inter_height = max(0, y_max_inter - y_min_inter)
        intersection = inter_width * inter_height
        
        # Calculate union area
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        # Calculate IoU
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return float(iou)
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode ('human' for display)
        """
        fig, ax = plt.subplots(1, figsize=(8, 8))
        
        # Display image
        ax.imshow(self.image)
        
        # Draw ground truth box (green)
        gt_rect = patches.Rectangle(
            (self.ground_truth_box[0], self.ground_truth_box[1]),
            self.ground_truth_box[2] - self.ground_truth_box[0],
            self.ground_truth_box[3] - self.ground_truth_box[1],
            linewidth=2,
            edgecolor='green',
            facecolor='none',
            label='Ground Truth'
        )
        ax.add_patch(gt_rect)
        
        # Draw current box (red)
        curr_rect = patches.Rectangle(
            (self.current_box[0], self.current_box[1]),
            self.current_box[2] - self.current_box[0],
            self.current_box[3] - self.current_box[1],
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            label='Current Box'
        )
        ax.add_patch(curr_rect)
        
        # Add IoU text
        iou = self._calculate_iou(self.current_box, self.ground_truth_box)
        ax.text(
            10, 30,
            f'IoU: {iou:.3f} | Step: {self.step_count}/{self.max_steps}',
            color='white',
            fontsize=12,
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7)
        )
        
        ax.legend(loc='upper right')
        ax.axis('off')
        plt.tight_layout()
        
        if mode == 'human':
            plt.show()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Create a sample image (256x256x3)
    sample_image = np.array(Image.open("tester.jpeg"))
    
    # Define ground truth box
    gt_box = (342, 271, 734, 644)
    
    # Initialize environment
    env = DQNObjectDetectionEnv(
        image=sample_image,
        ground_truth_box=gt_box,
        max_steps=15,
        move_percentage=0.1
    )
    
    # Reset and run a few random steps
    state = env.reset()
    print(f"Initial State: {state}")
    print(f"Initial IoU: {env.previous_iou:.3f}")
    
    for step in range(5):
        action = np.random.randint(0, 9)
        next_state, reward, done, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  IoU: {info['iou']:.3f}")
        print(f"  Done: {done}")
        
        if done:
            break
    
    # Render final state
    env.render()