import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from communication_env import ComNetEnv
from train_mappo import Actor, Critic
import json
import os
from typing import Dict, List


def load_model(checkpoint_path: str, obs_dim: int, state_dim: int, device: str = 'cpu'):
    """Load trained model from checkpoint."""
    actor = Actor(obs_dim=obs_dim).to(device)
    critic = Critic(state_dim=state_dim).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])

    actor.eval()
    critic.eval()

    return actor, critic, checkpoint.get('update', 0)


def analyze_policy(env, actor, num_episodes: int = 10, device: str = 'cpu'):
    """Analyze the learned policy behavior."""

    # Statistics to track
    action_counts = np.zeros((env.max_agents, 4))
    cam_rates = []
    cpm_rates = []
    knowledge_per_step = []
    bytes_per_step = []
    action_patterns = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_actions = []

        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                dist = actor(obs_tensor)
                actions = dist.sample().cpu().numpy()
                probs = dist.probs.cpu().numpy()

            # Track actions for valid agents only
            valid_actions = actions[:env.num_agents]
            episode_actions.append(valid_actions)

            # Update action counts
            for i in range(env.num_agents):
                action_counts[i, valid_actions[i]] += 1

            obs, rewards, done, info = env.step(actions)

            # Track metrics
            cam_sent = (valid_actions == 1).sum() + (valid_actions == 3).sum()
            cpm_sent = (valid_actions >= 2).sum()

            cam_rates.append(cam_sent / env.num_agents)
            cam_rates.append(cam_sent / env.num_agents)
            cpm_rates.append(cpm_sent / min(env.num_cpm_agents, env.num_agents))
            knowledge_per_step.append(info['total_knowledge'])
            bytes_per_step.append(info['total_bytes'])

        # Store as list since episode_actions may have variable lengths
        action_patterns.append(episode_actions)

    return {
        'action_counts': action_counts,
        'cam_rates': np.array(cam_rates),
        'cpm_rates': np.array(cpm_rates),
        'knowledge': np.array(knowledge_per_step),
        'bytes': np.array(bytes_per_step),
        'action_patterns': action_patterns
    }


def plot_training_curves(log_dir: str):
    """Plot training curves from saved logs."""
    # This would read from your logging system
    # For now, we'll create a placeholder
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Placeholder for actual data
    updates = np.arange(100)
    rewards = np.random.randn(100).cumsum() + 100
    knowledge = np.random.randn(100).cumsum() + 500
    bytes_sent = np.random.randn(100).cumsum() + 10000

    axes[0, 0].plot(updates, rewards)
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].set_xlabel('Update')

    axes[0, 1].plot(updates, knowledge)
    axes[0, 1].set_title('Total Knowledge')
    axes[0, 1].set_xlabel('Update')

    axes[1, 0].plot(updates, bytes_sent)
    axes[1, 0].set_title('Bytes Transmitted')
    axes[1, 0].set_xlabel('Update')

    axes[1, 1].axis('off')

    plt.tight_layout()
    return fig


def plot_policy_analysis(stats: Dict):
    """Visualize policy behavior."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Action distribution heatmap
    action_dist = stats['action_counts'][:100, :]  # First 100 agents
    action_dist = action_dist / action_dist.sum(axis=1, keepdims=True)

    sns.heatmap(action_dist.T, ax=axes[0, 0], cmap='YlOrRd',
                xticklabels=False, yticklabels=['None', 'CAM', 'CPM', 'Both'])
    axes[0, 0].set_title('Action Distribution by Agent')
    axes[0, 0].set_xlabel('Agent ID')

    # CAM/CPM rates over time
    axes[0, 1].plot(stats['cam_rates'], label='CAM Rate', alpha=0.7)
    axes[0, 1].plot(stats['cpm_rates'], label='CPM Rate', alpha=0.7)
    axes[0, 1].set_title('Communication Rates')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Fraction Sending')
    axes[0, 1].legend()

    # Knowledge vs Bytes tradeoff
    axes[0, 2].scatter(stats['bytes'], stats['knowledge'], alpha=0.5)
    axes[0, 2].set_title('Knowledge vs Bytes Tradeoff')
    axes[0, 2].set_xlabel('Bytes Transmitted')
    axes[0, 2].set_ylabel('Total Knowledge')

    # Average action distribution
    total_actions = stats['action_counts'].sum(axis=0)
    action_names = ['None', 'CAM', 'CPM', 'Both']
    axes[1, 0].bar(action_names, total_actions / total_actions.sum())
    axes[1, 0].set_title('Overall Action Distribution')
    axes[1, 0].set_ylabel('Fraction')

    # Knowledge efficiency
    knowledge_efficiency = stats['knowledge'] / (stats['bytes'] + 1)
    axes[1, 1].plot(knowledge_efficiency)
    axes[1, 1].set_title('Knowledge Efficiency (Knowledge/Byte)')
    axes[1, 1].set_xlabel('Step')

    # Temporal patterns
    if len(stats['action_patterns']) > 0:
        pattern_episode = stats['action_patterns'][0]  # First episode
        if len(pattern_episode) > 0:
            # Convert list of variable-length arrays to a padded matrix
            max_agents = max(len(step_actions) for step_actions in pattern_episode)
            max_steps = min(20, len(pattern_episode))
            max_agents_to_show = min(50, max_agents)

            # Create padded pattern matrix
            pattern_matrix = np.full((max_steps, max_agents_to_show), -1, dtype=int)
            for t, step_actions in enumerate(pattern_episode[:max_steps]):
                n_agents = min(len(step_actions), max_agents_to_show)
                pattern_matrix[t, :n_agents] = step_actions[:n_agents]

            # Mask invalid entries
            pattern_matrix = np.ma.masked_where(pattern_matrix == -1, pattern_matrix)

            sns.heatmap(pattern_matrix.T, ax=axes[1, 2], cmap='viridis',
                        cbar_kws={'label': 'Action'}, mask=(pattern_matrix.T == -1))
            axes[1, 2].set_title('Action Pattern (First Episode)')
            axes[1, 2].set_xlabel('Step')
            axes[1, 2].set_ylabel('Agent ID')

    plt.tight_layout()
    return fig


def compare_policies(env, actor_trained, num_episodes: int = 10, device: str = 'cpu'):
    """Compare trained policy with naive baseline."""
    from policies import naive_policy

    results = {
        'trained': {'rewards': [], 'knowledge': [], 'bytes': []},
        'naive': {'rewards': [], 'knowledge': [], 'bytes': []}
    }

    # Evaluate trained policy
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_knowledge = 0
        episode_bytes = 0

        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).to(device)
                dist = actor_trained(obs_tensor)
                actions = dist.sample().cpu().numpy()

            obs, rewards, done, info = env.step(actions)
            episode_reward += rewards[:env.num_agents].sum()
            episode_knowledge += info['total_knowledge']
            episode_bytes += info['total_bytes']

        results['trained']['rewards'].append(episode_reward)
        results['trained']['knowledge'].append(episode_knowledge)
        results['trained']['bytes'].append(episode_bytes)

    # Evaluate naive policy
    for _ in range(num_episodes):
        env.reset()
        done = False
        episode_reward = 0
        episode_knowledge = 0
        episode_bytes = 0
        step_idx = 0

        while not done:
            # Get naive actions
            is_send_cam, is_send_cpm = naive_policy(env.step_list[env.current_step])

            # Convert to action format
            actions = np.zeros(env.max_agents, dtype=int)
            actions[:env.num_agents] = 1  # Default CAM
            actions[:env.num_cpm_agents] = np.where(is_send_cpm[:env.num_cpm_agents], 3, 1)

            obs, rewards, done, info = env.step(actions)
            episode_reward += rewards[:env.num_agents].sum()
            episode_knowledge += info['total_knowledge']
            episode_bytes += info['total_bytes']

        results['naive']['rewards'].append(episode_reward)
        results['naive']['knowledge'].append(episode_knowledge)
        results['naive']['bytes'].append(episode_bytes)

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['rewards', 'knowledge', 'bytes']
    titles = ['Total Reward', 'Total Knowledge', 'Total Bytes']

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        trained_mean = np.mean(results['trained'][metric])
        trained_std = np.std(results['trained'][metric])
        naive_mean = np.mean(results['naive'][metric])
        naive_std = np.std(results['naive'][metric])

        x = ['Naive', 'Trained']
        y = [naive_mean, trained_mean]
        err = [naive_std, trained_std]

        axes[i].bar(x, y, yerr=err, capsize=10)
        axes[i].set_title(title)
        axes[i].set_ylabel('Value')

        # Add improvement percentage
        if metric == 'bytes':
            improvement = (naive_mean - trained_mean) / naive_mean * 100
            axes[i].text(0.5, max(y) * 1.1, f'{improvement:.1f}% reduction',
                         ha='center', fontsize=10, color='green')
        else:
            improvement = (trained_mean - naive_mean) / naive_mean * 100
            axes[i].text(0.5, max(y) * 1.1, f'{improvement:.1f}% improvement',
                         ha='center', fontsize=10, color='green')

    plt.tight_layout()
    return fig, results


def main():
    """Main analysis script."""
    # Configuration
    model_dir = "mappo_runs/20250610_151019"  # Update with your actual directory
    checkpoint = "best_model.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load config
    with open(f"{model_dir}/config.json", 'r') as f:
        config = json.load(f)

    # Create test environment
    test_steps = list(range(800, 1000))  # Adjust based on your data
    test_env = ComNetEnv(
        max_agents=config['max_agents'],
        step_list=test_steps,
        alpha=1.0,
        beta=0.001,
        normalize_reward=True,
        share_reward=True,
        obs_include_global=True
    )

    # Get dimensions
    obs_dim = test_env.observation_space.shape[0]
    # Reset environment first to initialize state variables
    test_env.reset()
    state_dim = len(test_env.get_state())

    # Load model
    print(f"Loading model from {model_dir}/{checkpoint}")
    actor, critic, update = load_model(
        f"{model_dir}/{checkpoint}",
        obs_dim, state_dim, device
    )
    print(f"Model loaded from update {update}")

    # Analyze policy
    print("\nAnalyzing policy behavior...")
    stats = analyze_policy(test_env, actor, num_episodes=10, device=device)

    # Create visualizations
    print("Creating visualizations...")

    # Policy analysis plots
    fig1 = plot_policy_analysis(stats)
    plt.savefig(f"{model_dir}/policy_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Compare with baseline
    print("\nComparing with naive baseline...")
    fig2, comparison_results = compare_policies(test_env, actor, num_episodes=10, device=device)
    plt.savefig(f"{model_dir}/policy_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Average CAM rate: {stats['cam_rates'].mean():.3f} ± {stats['cam_rates'].std():.3f}")
    print(f"Average CPM rate: {stats['cpm_rates'].mean():.3f} ± {stats['cpm_rates'].std():.3f}")
    print(f"Average knowledge per step: {stats['knowledge'].mean():.1f} ± {stats['knowledge'].std():.1f}")
    print(f"Average bytes per step: {stats['bytes'].mean():.0f} ± {stats['bytes'].std():.0f}")

    print("\n=== Comparison Results ===")
    for policy in ['naive', 'trained']:
        print(f"\n{policy.capitalize()} Policy:")
        for metric in ['rewards', 'knowledge', 'bytes']:
            values = comparison_results[policy][metric]
            print(f"  {metric}: {np.mean(values):.1f} ± {np.std(values):.1f}")


if __name__ == "__main__":
    main()