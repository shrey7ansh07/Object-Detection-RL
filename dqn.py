import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt


# Experience replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    """Experience Replay Buffer for storing and sampling transitions."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward, done):
        """Store a transition in the buffer."""
        self.buffer.append(Transition(state, action, next_state, reward, done))
    
    def sample(self, batch_size: int):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            batch: List of sampled transitions
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


class DQNetwork(nn.Module):
    """Deep Q-Network for approximating Q-values."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128, 64]):
        """
        Initialize DQN.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: List of hidden layer dimensions
        """
        super(DQNetwork, self).__init__()
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """
        Forward pass through network.
        
        Args:
            state: Input state tensor
            
        Returns:
            q_values: Q-values for all actions
        """
        return self.network(state)


class DQNAgent:
    """DQN Agent for Object Detection Task."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        device: str = None
    ):
        """
        Initialize DQN Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_capacity: Replay buffer capacity
            batch_size: Batch size for training
            target_update_freq: Frequency to update target network
            device: Device to run on ('cuda' or 'cpu')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device configuration
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize networks
        self.policy_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_capacity)
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.episode_ious = []
        self.losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (uses epsilon-greedy)
            
        Returns:
            action: Selected action index
        """
        if training and random.random() < self.epsilon:
            # Random action (exploration)
            return random.randint(0, self.action_dim - 1)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def store_transition(self, state, action, next_state, reward, done):
        """Store transition in replay buffer."""
        self.memory.push(state, action, next_state, reward, done)
    
    def train_step(self):
        """
        Perform one training step (update policy network).
        
        Returns:
            loss: Training loss value
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        
        # Compute target Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_step += 1
        
        # Update target network periodically
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def update_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath: str):
        """Save model weights."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        print(f"Model loaded from {filepath}")


def train_dqn(env, agent, num_episodes: int = 1000, verbose: bool = True):
    """
    Train the DQN agent.
    
    Args:
        env: DQNObjectDetectionEnv instance
        agent: DQNAgent instance
        num_episodes: Number of training episodes
        verbose: Whether to print training progress
        
    Returns:
        training_stats: Dictionary containing training statistics
    """
    episode_rewards = []
    episode_ious = []
    episode_lengths = []
    losses = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        
        while not done:
            # Select and perform action
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, next_state, reward, done)
            
            # Train agent
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
        
        # Decay epsilon
        agent.update_epsilon()
        
        # Record statistics
        episode_rewards.append(episode_reward)
        episode_ious.append(info['iou'])
        episode_lengths.append(info['step'])
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Print progress
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_iou = np.mean(episode_ious[-50:])
            avg_loss = np.mean(losses[-50:]) if losses[-50:] else 0
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg IoU: {avg_iou:.3f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Avg Loss: {avg_loss:.4f}")
    
    training_stats = {
        'episode_rewards': episode_rewards,
        'episode_ious': episode_ious,
        'episode_lengths': episode_lengths,
        'losses': losses
    }
    
    return training_stats


def evaluate_dqn(env, agent, num_episodes: int = 10, render: bool = False):
    """
    Evaluate the trained DQN agent.
    
    Args:
        env: DQNObjectDetectionEnv instance
        agent: DQNAgent instance
        num_episodes: Number of evaluation episodes
        render: Whether to render episodes
        
    Returns:
        eval_stats: Dictionary containing evaluation statistics
    """
    eval_rewards = []
    eval_ious = []
    eval_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            
            if render:
                env.render()
        
        eval_rewards.append(episode_reward)
        eval_ious.append(info['iou'])
        eval_lengths.append(info['step'])
        
        print(f"Eval Episode {episode + 1}: Reward={episode_reward:.2f}, IoU={info['iou']:.3f}, Steps={info['step']}")
    
    eval_stats = {
        'rewards': eval_rewards,
        'ious': eval_ious,
        'lengths': eval_lengths,
        'avg_reward': np.mean(eval_rewards),
        'avg_iou': np.mean(eval_ious),
        'avg_length': np.mean(eval_lengths)
    }
    
    return eval_stats


def plot_training_stats(training_stats, save_path: str = None):
    """
    Plot training statistics.
    
    Args:
        training_stats: Dictionary containing training statistics
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot episode rewards
    axes[0, 0].plot(training_stats['episode_rewards'], alpha=0.6)
    axes[0, 0].plot(np.convolve(training_stats['episode_rewards'], 
                                 np.ones(50)/50, mode='valid'), 
                    linewidth=2, label='Moving Avg (50)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot episode IoUs
    axes[0, 1].plot(training_stats['episode_ious'], alpha=0.6)
    axes[0, 1].plot(np.convolve(training_stats['episode_ious'], 
                                 np.ones(50)/50, mode='valid'), 
                    linewidth=2, label='Moving Avg (50)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Final IoU')
    axes[0, 1].set_title('Episode IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot losses
    if training_stats['losses']:
        axes[1, 0].plot(training_stats['losses'], alpha=0.6)
        axes[1, 0].plot(np.convolve(training_stats['losses'], 
                                     np.ones(50)/50, mode='valid'), 
                        linewidth=2, label='Moving Avg (50)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    axes[1, 1].plot(training_stats['episode_lengths'], alpha=0.6)
    axes[1, 1].plot(np.convolve(training_stats['episode_lengths'], 
                                 np.ones(50)/50, mode='valid'), 
                    linewidth=2, label='Moving Avg (50)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Episode Length')
    axes[1, 1].set_title('Episode Length')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":
    # Note: This requires the DQNObjectDetectionEnv class from the previous artifact
    
    # Create sample environment
    sample_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    gt_box = (80, 80, 180, 180)
    
    from env import DQNObjectDetectionEnv  # Import the environment
    
    env = DQNObjectDetectionEnv(
        image=sample_image,
        ground_truth_box=gt_box,
        max_steps=15,
        move_percentage=0.1
    )
    
    # Initialize DQN agent
    state_dim = 5  # [x_min, y_min, x_max, y_max, iou]
    action_dim = 9  # 9 discrete actions
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=10
    )
    
    print("Starting training...")
    print(f"Device: {agent.device}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print("-" * 50)
    
    # Train the agent
    training_stats = train_dqn(env, agent, num_episodes=500, verbose=True)
    
    # Save the model
    agent.save_model("dqn_object_detection.pth")
    
    # Plot training statistics
    plot_training_stats(training_stats, save_path="training_stats.png")
    
    # Evaluate the agent
    print("\n" + "=" * 50)
    print("Evaluating trained agent...")
    print("=" * 50)
    eval_stats = evaluate_dqn(env, agent, num_episodes=10, render=False)
    
    print("\nEvaluation Results:")
    print(f"Average Reward: {eval_stats['avg_reward']:.2f}")
    print(f"Average IoU: {eval_stats['avg_iou']:.3f}")
    print(f"Average Episode Length: {eval_stats['avg_length']:.1f}")