import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Dict, List
import os
from datetime import datetime
import json

# Import your environment
from communication_env import ComNetEnv
from tqdm import tqdm

class Actor(nn.Module):
    """Shared actor network for all agents."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256, action_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return Categorical(logits=logits)


class Critic(nn.Module):
    """Centralized critic network."""

    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class RolloutBuffer:
    """Buffer to store trajectories."""

    def __init__(self, max_agents: int, obs_dim: int, state_dim: int, rollout_length: int):
        self.max_agents = max_agents
        self.rollout_length = rollout_length

        # Pre-allocate buffers
        self.obs = np.zeros((rollout_length, max_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_length, max_agents), dtype=np.int64)
        self.rewards = np.zeros((rollout_length, max_agents), dtype=np.float32)
        self.log_probs = np.zeros((rollout_length, max_agents), dtype=np.float32)
        self.values = np.zeros((rollout_length + 1,), dtype=np.float32)
        self.states = np.zeros((rollout_length + 1, state_dim), dtype=np.float32)
        self.masks = np.zeros((rollout_length, max_agents), dtype=np.float32)
        self.dones = np.zeros((rollout_length,), dtype=np.float32)

        self.ptr = 0

    def add(self, obs, actions, rewards, log_probs, value, state, mask, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = value
        self.states[self.ptr] = state
        self.masks[self.ptr] = mask
        self.dones[self.ptr] = done
        self.ptr += 1

    def get(self):
        return {
            'obs': self.obs,
            'actions': self.actions,
            'rewards': self.rewards,
            'log_probs': self.log_probs,
            'values': self.values[:-1],
            'states': self.states[:-1],
            'masks': self.masks,
            'dones': self.dones
        }


def compute_gae(rewards, values, dones, masks, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation.

    Parameters:
    -----------
    rewards: (T, N) rewards for each timestep and agent
    values: (T+1,) value estimates (centralized)
    dones: (T,) episode termination flags
    masks: (T, N) agent validity masks
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    returns = np.zeros((T, N), dtype=np.float32)

    # Sum rewards across agents for centralized value
    summed_rewards = (rewards * masks).sum(axis=1)  # (T,)

    # Compute advantages backwards
    gae = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = summed_rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae

        # Broadcast advantage to all agents
        advantages[t] = gae
        returns[t] = gae + values[t]

    # Mask invalid agents
    advantages = advantages * masks
    returns = returns * masks

    return advantages, returns


def collect_rollout(env, actor, critic, buffer, rollout_length, device):
    """Optimized rollout collection."""
    obs = env.reset()
    state = env.get_state()

    # Batch initial value computation
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        value = critic(state_tensor).cpu().numpy().flatten()[0]

    buffer.states[0] = state
    buffer.values[0] = value

    for step in range(rollout_length):
        # Create mask once
        mask = np.zeros(env.max_agents, dtype=np.float32)
        mask[:env.num_agents] = 1.0

        # Batch action sampling - only process active agents
        with torch.no_grad():
            active_obs = obs[:env.num_agents]
            obs_tensor = torch.FloatTensor(active_obs).to(device)
            dist = actor(obs_tensor)
            active_actions = dist.sample()
            active_log_probs = dist.log_prob(active_actions)

            # Pad to full size
            actions = np.zeros(env.max_agents, dtype=np.int64)
            log_probs = np.zeros(env.max_agents, dtype=np.float32)

            actions[:env.num_agents] = active_actions.cpu().numpy()
            log_probs[:env.num_agents] = active_log_probs.cpu().numpy()

        # Step environment
        next_obs, rewards, done, info = env.step(actions)
        next_state = env.get_state()

        # Get next value
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            next_value = critic(next_state_tensor).cpu().numpy().flatten()[0]

        # Store in buffer
        buffer.add(obs, actions, rewards, log_probs, value, state, mask, done)

        # Update for next iteration
        obs = next_obs
        state = next_state
        value = next_value

        if done:
            obs = env.reset()
            state = env.get_state()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                value = critic(state_tensor).cpu().numpy().flatten()[0]

    # Store final value and state
    buffer.values[rollout_length] = value
    buffer.states[rollout_length] = state

    return buffer


def ppo_update(actor, critic, actor_optimizer, critic_optimizer, data,
               clip_ratio=0.2, value_clip=0.2, entropy_coef=0.01,
               num_epochs=4, batch_size=64, device='cuda'):
    """PPO update step."""

    # Convert to tensors
    obs = torch.FloatTensor(data['obs']).to(device)
    actions = torch.LongTensor(data['actions']).to(device)
    old_log_probs = torch.FloatTensor(data['log_probs']).to(device)
    states = torch.FloatTensor(data['states']).to(device)
    masks = torch.FloatTensor(data['masks']).to(device)

    # Flatten for minibatch sampling
    T, N = obs.shape[:2]
    obs_flat = obs.reshape(T * N, -1)
    actions_flat = actions.reshape(T * N)
    old_log_probs_flat = old_log_probs.reshape(T * N)
    masks_flat = masks.reshape(T * N)

    # Pre-filter valid indices
    valid_indices = torch.where(masks_flat > 0)[0]
    if len(valid_indices) == 0:
        return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}

    # Work only with valid data
    obs_valid = obs_flat[valid_indices]
    actions_valid = actions_flat[valid_indices]
    old_log_probs_valid = old_log_probs_flat[valid_indices]

    # Compute advantages and returns
    advantages, returns = compute_gae(
        data['rewards'], data['values'], data['dones'], data['masks']
    )

    # Normalize advantages
    valid_advantages = advantages[data['masks'] > 0]
    if len(valid_advantages) > 1:
        adv_mean = valid_advantages.mean()
        adv_std = valid_advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

    advantages = torch.FloatTensor(advantages).to(device)
    returns = torch.FloatTensor(returns).to(device)
    advantages_flat = advantages.reshape(T * N)
    advantages_valid = advantages_flat[valid_indices]

    # Pre-compute returns for critic
    returns_mean = returns.mean(dim=1)

    # Training loop
    num_valid = len(valid_indices)
    for epoch in range(num_epochs):
        # Generate random permutation of valid indices only
        perm = torch.randperm(num_valid)

        for start in range(0, num_valid, batch_size):
            end = min(start + batch_size, num_valid)
            batch_indices = perm[start:end]

            # Actor loss - work with pre-filtered data
            dist = actor(obs_valid[batch_indices])
            new_log_probs = dist.log_prob(actions_valid[batch_indices])
            entropy = dist.entropy()

            # Importance sampling ratio
            ratio = torch.exp(new_log_probs - old_log_probs_valid[batch_indices])

            # Clipped surrogate objective
            batch_advantages = advantages_valid[batch_indices]
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * batch_advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -entropy_coef * entropy.mean()

            # Critic loss - batch unique states
            original_indices = valid_indices[batch_indices]
            state_indices = original_indices // N
            unique_states, inverse_indices = torch.unique(state_indices, return_inverse=True)

            values = critic(states[unique_states]).squeeze()
            value_targets = returns_mean[unique_states]

            critic_loss = F.mse_loss(values, value_targets)

            # Update networks
            actor_optimizer.zero_grad()
            (actor_loss + entropy_loss).backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_optimizer.step()

    return {
        'actor_loss': actor_loss.item(),
        'critic_loss': critic_loss.item(),
        'entropy': entropy_loss.item() / entropy_coef
    }


def evaluate(env, actor, num_episodes=5, device='cuda'):
    """Evaluate the policy."""
    total_rewards = []
    total_knowledge = []
    total_bytes = []

    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_knowledge = 0
        episode_bytes = 0
        done = False

        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                dist = actor(obs_tensor)
                actions = dist.sample().cpu().numpy()

            obs, rewards, done, info = env.step(actions)

            # Only count rewards for valid agents
            valid_rewards = rewards[:env.num_agents]
            episode_reward += valid_rewards.sum()
            episode_knowledge += info['total_knowledge']
            episode_bytes += info['total_bytes']

        total_rewards.append(episode_reward)
        total_knowledge.append(episode_knowledge)
        total_bytes.append(episode_bytes)

    return {
        'mean_reward': np.mean(total_rewards),
        'mean_knowledge': np.mean(total_knowledge),
        'mean_bytes': np.mean(total_bytes),
        'std_reward': np.std(total_rewards)
    }


def main():
    # Configuration
    config = {
        'max_agents': 600,
        'rollout_length': 128,
        'total_timesteps': 12_800,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'entropy_coef': 0.01,
        'num_ppo_epochs': 4,
        'batch_size': 64,
        'eval_interval': 10,
        'save_interval': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"mappo_runs/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    with open(f"{save_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=4)

    # Split data into train/test
    total_steps = 12000  # Adjust based on your data
    train_split = 0.8
    train_steps = list(range(1, int(total_steps * train_split) + 1))
    test_steps = list(range(int(total_steps * train_split)+1, total_steps+1))

    # Create environments
    train_env = ComNetEnv(
        max_agents=config['max_agents'],
        step_list=train_steps,
        alpha=1,
        beta=1,
        normalize_reward=True,
        share_reward=True,
        obs_include_global=True
    )

    test_env = ComNetEnv(
        max_agents=config['max_agents'],
        step_list=test_steps,
        alpha=1,
        beta=1,
        normalize_reward=True,
        share_reward=True,
        obs_include_global=True
    )

    # Get dimensions
    obs_dim = train_env.observation_space.shape[0]
    # Reset environment first to initialize state variables
    train_env.reset()
    state_dim = len(train_env.get_state())

    # Initialize models
    actor = Actor(obs_dim=obs_dim).to(config['device'])
    critic = Critic(state_dim=state_dim).to(config['device'])

    actor_optimizer = optim.Adam(actor.parameters(), lr=config['learning_rate'])
    critic_optimizer = optim.Adam(critic.parameters(), lr=config['learning_rate'])

    # Create buffer
    buffer = RolloutBuffer(
        max_agents=config['max_agents'],
        obs_dim=obs_dim,
        state_dim=state_dim,
        rollout_length=config['rollout_length']
    )

    # Training loop
    num_updates = config['total_timesteps'] // config['rollout_length']
    best_test_reward = -float('inf')

    print(f"Starting MAPPO training...")
    print(f"Total updates: {num_updates}")
    print(f"Device: {config['device']}")
    print(f"Save directory: {save_dir}")

    for update in tqdm(range(num_updates), desc="Training MAPPO", unit="update"):
        # Collect rollout
        buffer.ptr = 0  # Reset buffer
        collect_rollout(train_env, actor, critic, buffer,
                        config['rollout_length'], config['device'])

        # Get data
        data = buffer.get()

        # PPO update
        update_stats = ppo_update(
            actor, critic, actor_optimizer, critic_optimizer, data,
            clip_ratio=config['clip_ratio'],
            entropy_coef=config['entropy_coef'],
            num_epochs=config['num_ppo_epochs'],
            batch_size=config['batch_size'],
            device=config['device']
        )

        # Evaluation
        if update % config['eval_interval'] == 0:
            train_stats = evaluate(train_env, actor, num_episodes=3, device=config['device'])
            test_stats = evaluate(test_env, actor, num_episodes=5, device=config['device'])

            print(f"\nEvaluation at update {update}:")
            print(f"  Train - Reward: {train_stats['mean_reward']:.2f} ± {train_stats['std_reward']:.2f}")
            print(f"  Train - Knowledge: {train_stats['mean_knowledge']:.2f}, Bytes: {train_stats['mean_bytes']:.0f}")
            print(f"  Test  - Reward: {test_stats['mean_reward']:.2f} ± {test_stats['std_reward']:.2f}")
            print(f"  Test  - Knowledge: {test_stats['mean_knowledge']:.2f}, Bytes: {test_stats['mean_bytes']:.0f}")

            # Save best model
            if test_stats['mean_reward'] > best_test_reward:
                best_test_reward = test_stats['mean_reward']
                torch.save({
                    'actor_state_dict': actor.state_dict(),
                    'critic_state_dict': critic.state_dict(),
                    'update': update,
                    'test_reward': best_test_reward
                }, f"{save_dir}/best_model.pt")
                print(f"  New best model saved!")

        # Regular checkpoint
        if update % config['save_interval'] == 0:
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'actor_optimizer': actor_optimizer.state_dict(),
                'critic_optimizer': critic_optimizer.state_dict(),
                'update': update
            }, f"{save_dir}/checkpoint_{update}.pt")

    print("\nTraining completed!")
    print(f"Best test reward: {best_test_reward:.2f}")


if __name__ == "__main__":
    main()