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


def plot_policy_analysis(stats: Dict, out_dir: str):
    """Visualize and save selected plots of policy behavior."""
    os.makedirs(out_dir, exist_ok=True)

    # 1. Action distribution heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    action_dist = stats['action_counts'][:100, :]
    action_dist = action_dist / action_dist.sum(axis=1, keepdims=True)
    sns.heatmap(action_dist.T, ax=ax, cmap='YlOrRd', xticklabels=False, yticklabels=['None', 'CAM', 'CPM', 'Both'])
    ax.set_title('Action Distribution by Agent')
    ax.set_xlabel('Agent ID')
    fig.tight_layout()
    fig.savefig(f"{out_dir}/action_distribution.png", dpi=150)
    plt.close(fig)

    # 2. Knowledge vs Bytes tradeoff
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(stats['bytes'], stats['knowledge'], alpha=0.5)
    ax.set_title('Knowledge vs Bytes Tradeoff')
    ax.set_xlabel('Bytes Transmitted')
    ax.set_ylabel('Total Knowledge')
    fig.tight_layout()
    fig.savefig(f"{out_dir}/knowledge_vs_bytes.png", dpi=150)
    plt.close(fig)

    # 3. Overall action distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    total_actions = stats['action_counts'].sum(axis=0)
    action_names = ['None', 'CAM', 'CPM', 'Both']
    ax.bar(action_names, total_actions / total_actions.sum())
    ax.set_title('Overall Action Distribution')
    ax.set_ylabel('Fraction')
    fig.tight_layout()
    fig.savefig(f"{out_dir}/overall_action_distribution.png", dpi=150)
    plt.close(fig)

    # 4. Temporal patterns (first episode)
    if len(stats['action_patterns']) > 0 and len(stats['action_patterns'][0]) > 0:
        pattern_episode = stats['action_patterns'][0]
        max_agents = max(len(step_actions) for step_actions in pattern_episode)
        max_steps = min(20, len(pattern_episode))
        max_agents_to_show = min(50, max_agents)

        pattern_matrix = np.full((max_steps, max_agents_to_show), -1, dtype=int)
        for t, step_actions in enumerate(pattern_episode[:max_steps]):
            n_agents = min(len(step_actions), max_agents_to_show)
            pattern_matrix[t, :n_agents] = step_actions[:n_agents]
        pattern_matrix = np.ma.masked_where(pattern_matrix == -1, pattern_matrix)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pattern_matrix.T, ax=ax, cmap='viridis', cbar_kws={'label': 'Action'},
                    mask=(pattern_matrix.T == -1))
        ax.set_title('Action Pattern (First Episode)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Agent ID')
        fig.tight_layout()
        fig.savefig(f"{out_dir}/temporal_action_pattern.png", dpi=150)
        plt.close(fig)




def compare_policies(env, actor_trained, num_episodes: int = 10, device: str = 'cpu', out_dir: str = './output'):
    """Compare trained policy with naive baseline and save separate plots."""
    from policies import naive_policy
    os.makedirs(out_dir, exist_ok=True)

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

        while not done:
            step = env.step_list[env.current_step]
            is_send_cam, is_send_cpm = naive_policy(step)
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

    # Plot each comparison individually
    metrics = ['rewards', 'knowledge', 'bytes']
    titles = ['Total Reward', 'Total Knowledge', 'Total Bytes']
    filenames = ['comparison_reward.png', 'comparison_knowledge.png', 'comparison_bytes.png']

    for metric, title, fname in zip(metrics, titles, filenames):
        fig, ax = plt.subplots(figsize=(6, 4))
        trained_mean = np.mean(results['trained'][metric])
        trained_std = np.std(results['trained'][metric])
        naive_mean = np.mean(results['naive'][metric])
        naive_std = np.std(results['naive'][metric])

        x = ['Naive', 'Trained']
        y = [naive_mean, trained_mean]
        err = [naive_std, trained_std]

        ax.bar(x, y, yerr=err, capsize=10)
        ax.set_title(title)
        ax.set_ylabel('Value')

        # Add improvement annotation
        if metric == 'bytes':
            improvement = (naive_mean - trained_mean) / naive_mean * 100
            sign = "reduction"
            color = "green"
        else:
            improvement = (trained_mean - naive_mean) / naive_mean * 100
            sign = "improvement"
            color = "green" if improvement >= 0 else "red"

        # Position text above the taller bar
        top_y = max(y) + max(err) * 1.2
        ax.text(0.5, top_y, f"{improvement:+.1f}% {sign}", ha='center', fontsize=10, color=color)


        fig.tight_layout()
        fig.savefig(f"{out_dir}/{fname}", dpi=150)
        plt.close(fig)

    return results


def main():
    """Main analysis script."""
    import datetime

    # Configuration
    model_dir = "mappo_runs/20250619_190642"  # Update with your actual directory
    checkpoint = "best_model.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load config
    with open(f"{model_dir}/config.json", 'r') as f:
        config = json.load(f)

    alpha = config.get('alpha', 1.0)
    beta = config.get('beta', 0.001)

    # Create test environment
    test_steps = list(range(800, 1000))  # Adjust based on your data
    test_env = ComNetEnv(
        max_agents=config['max_agents'],
        step_list=test_steps,
        alpha=alpha,
        beta=beta,
        normalize_reward=True,
        share_reward=True,
        obs_include_global=True
    )

    # Get dimensions
    obs_dim = test_env.observation_space.shape[0]
    test_env.reset()
    state_dim = len(test_env.get_state())

    # Load model
    print(f"Loading model from {model_dir}/{checkpoint}")
    actor, critic, update = load_model(
        f"{model_dir}/{checkpoint}",
        obs_dim, state_dim, device
    )
    print(f"Model loaded from update {update}")

    # Output directory
    # Output directory based on alpha and beta values
    out_dir = f"output/analyze_alpha{alpha}_beta{beta}"
    os.makedirs(out_dir, exist_ok=True)

    # Analyze policy
    print("\nAnalyzing policy behavior...")
    stats = analyze_policy(test_env, actor, num_episodes=10, device=device)

    # Create visualizations
    print("Creating visualizations...")
    plot_policy_analysis(stats, out_dir)

    print("\nComparing with naive baseline...")
    comparison_results = compare_policies(test_env, actor, num_episodes=10, device=device, out_dir=out_dir)

    # Summary statistics
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