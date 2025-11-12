import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from components import Actor, Critic, ReplayBuffer, Transition
import numpy as np



# ============================================================================
# TD3 AGENT
# ============================================================================

class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient Agent
    
    Key features:
    - Continuous actions (better for box adjustments)
    - Twin critics (reduces Q-overestimation)
    - Delayed policy updates (more stable)
    - Target policy smoothing (reduces variance)
    """
    
    def __init__(
        self,
        state_dim,
        action_dim=4,  # [dx, dy, dw, dh]
        device=torch.device('cpu'),
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.95,
        tau=0.005,  # Soft update rate
        policy_noise=0.2,  # Noise for target policy smoothing
        noise_clip=0.5,
        policy_freq=2,  # Delay policy updates
        buffer_size=100000,
        max_action=1.0
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.max_action = max_action
        
        # Networks
        self.actor = Actor(state_dim, action_dim, max_action=max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action=max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training tracking
        self.total_it = 0
        self.actor_losses = []
        self.critic_losses = []
        
        # Exploration noise (decays over time)
        self.exploration_noise = 0.1
        self.exploration_noise_decay = 0.9999
        self.exploration_noise_min = 0.01
    
    def act(self, state, add_noise=True):
        """
        Select action given state
        
        Args:
            state: Current state
            add_noise: Whether to add exploration noise (True for training)
        
        Returns:
            action: Continuous action [dx, dy, dw, dh]
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        # Add exploration noise during training
        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def train_step(self, batch_size=128):
        """
        Perform one TD3 training step
        
        Returns:
            Dictionary with loss values
        """
        if len(self.memory) < batch_size:
            return None
        
        self.total_it += 1
        
        # Sample batch
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.FloatTensor(np.array(batch.action)).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        with torch.no_grad():
            # Select action according to target policy with added noise
            noise = (torch.randn_like(action_batch) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state_batch) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Compute target Q-values (use minimum of two critics)
            target_Q1, target_Q2 = self.critic_target(next_state_batch, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + (1 - done_batch) * self.gamma * target_Q.squeeze()
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state_batch, action_batch)
        current_Q1 = current_Q1.squeeze()
        current_Q2 = current_Q2.squeeze()
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        self.critic_losses.append(critic_loss.item())
        
        # Delayed policy updates
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state_batch, self.actor(state_batch)).mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            self.actor_losses.append(actor_loss.item())
            
            # Soft update target networks
            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor, self.actor_target)
        
        # Decay exploration noise
        self.exploration_noise = max(
            self.exploration_noise_min,
            self.exploration_noise * self.exploration_noise_decay
        )
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else None,
            'exploration_noise': self.exploration_noise
        }
    
    def _soft_update(self, source, target):
        """Soft update target network parameters"""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        """Save agent state"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_it': self.total_it,
            'exploration_noise': self.exploration_noise,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses
        }, filepath)
    
    def load(self, filepath):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_it = checkpoint.get('total_it', 0)
        self.exploration_noise = checkpoint.get('exploration_noise', 0.1)
        self.actor_losses = checkpoint.get('actor_losses', [])
        self.critic_losses = checkpoint.get('critic_losses', [])
