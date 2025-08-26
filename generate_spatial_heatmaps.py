import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import imageio.v2 as imageio  # Use v2 explicitly to avoid deprecation warning
import os
from typing import Dict, List, Tuple
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from PIL import Image  # For image resizing if needed

from communication_env import ComNetEnv
from train_mappo import Actor
from matrix_utils import nrows_communication_matrix, nrows_vision_matrix


def load_position_data(step: int) -> Tuple[np.ndarray, List[str]]:
    """Load position data for a given step."""
    filepath = f"data/position/step_{step}.npz"

    try:
        data = np.load(filepath)
        positions = data["positions"]  # shape (n, 2)
        entity_ids = data["entity_ids"].tolist() if "entity_ids" in data else []

        # Validate positions
        if len(positions) == 0:
            raise ValueError("Empty positions array")

        # Check for NaN or Inf values
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            print(f"Warning: Invalid position values found at step {step}, cleaning...")
            # Replace invalid values with mean position
            valid_mask = ~(np.isnan(positions).any(axis=1) | np.isinf(positions).any(axis=1))
            if np.any(valid_mask):
                mean_pos = positions[valid_mask].mean(axis=0)
                positions[~valid_mask] = mean_pos
            else:
                # If all invalid, use origin
                positions = np.zeros_like(positions)

        return positions, entity_ids

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Warning: Issue loading positions for step {step}: {e}")
        # Return random positions as fallback
        try:
            n_agents = nrows_communication_matrix(step)
        except:
            n_agents = 0

        if n_agents == 0:
            n_agents = 100  # Default fallback

        positions = np.random.randn(n_agents, 2) * 100
        entity_ids = [f"agent_{i}" for i in range(n_agents)]
        return positions, entity_ids


def create_spatial_heatmap(positions: np.ndarray,
                           actions: np.ndarray,
                           grid_size: int = 100,
                           sigma: float = 5.0) -> Dict[str, np.ndarray]:
    """
    Create spatial heatmaps for different message types.

    Parameters
    ----------
    positions : np.ndarray
        Agent positions, shape (n_agents, 2)
    actions : np.ndarray
        Agent actions (0=none, 1=CAM, 2=CPM, 3=both)
    grid_size : int
        Resolution of the heatmap grid
    sigma : float
        Gaussian smoothing parameter

    Returns
    -------
    dict
        Dictionary with heatmaps for 'cam', 'cpm', and 'total' messages
    """
    # Validate inputs
    if len(positions) == 0:
        print("Warning: Empty positions array, creating default heatmap")
        positions = np.array([[0, 0]])
        actions = np.array([0])

    # Ensure actions match positions length
    if len(actions) > len(positions):
        actions = actions[:len(positions)]
    elif len(actions) < len(positions):
        actions = np.pad(actions, (0, len(positions) - len(actions)), 'constant', constant_values=0)

    # Get bounds with padding
    x_min, y_min = positions.min(axis=0) - 10
    x_max, y_max = positions.max(axis=0) + 10

    # Ensure valid bounds
    if np.isnan(x_min) or np.isnan(x_max) or np.isinf(x_min) or np.isinf(x_max):
        print("Warning: Invalid position bounds, using defaults")
        x_min, y_min = -100, -100
        x_max, y_max = 100, 100

    # Ensure non-zero range
    if x_max <= x_min:
        x_max = x_min + 1
    if y_max <= y_min:
        y_max = y_min + 1

    # Create grid
    x_bins = np.linspace(x_min, x_max, grid_size)
    y_bins = np.linspace(y_min, y_max, grid_size)

    # Initialize heatmaps
    heatmaps = {
        'cam': np.zeros((grid_size, grid_size)),
        'cpm': np.zeros((grid_size, grid_size)),
        'total': np.zeros((grid_size, grid_size))
    }

    # Count messages at each position
    for pos, action in zip(positions, actions):
        # Skip invalid positions
        if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
            continue

        # Find grid cell
        x_idx = np.digitize(pos[0], x_bins) - 1
        y_idx = np.digitize(pos[1], y_bins) - 1

        if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
            # CAM messages (actions 1 and 3)
            if action in [1, 3]:
                heatmaps['cam'][y_idx, x_idx] += 1

            # CPM messages (actions 2 and 3)
            if action in [2, 3]:
                heatmaps['cpm'][y_idx, x_idx] += 1

            # Any message
            if action > 0:
                heatmaps['total'][y_idx, x_idx] += 1

    # Apply Gaussian smoothing
    for key in heatmaps:
        if np.any(heatmaps[key] > 0):  # Only smooth if there's data
            heatmaps[key] = gaussian_filter(heatmaps[key], sigma=sigma)

    # Store grid info for plotting
    heatmaps['extent'] = [float(x_min), float(x_max), float(y_min), float(y_max)]
    heatmaps['positions'] = positions
    heatmaps['actions'] = actions

    return heatmaps


def plot_heatmap(heatmap_data: Dict[str, np.ndarray],
                 step: int,
                 message_type: str = 'total',
                 save_path: str = None,
                 show_agents: bool = True):
    """Plot a single heatmap with agent positions."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create custom colormap
    if message_type == 'cam':
        cmap = 'Blues'
        title = f'CAM Message Density - Step {step}'
    elif message_type == 'cpm':
        cmap = 'Reds'
        title = f'CPM Message Density - Step {step}'
    else:
        cmap = 'viridis'
        title = f'Total Message Density - Step {step}'

    # Plot heatmap
    im = ax.imshow(heatmap_data[message_type],
                   cmap=cmap,
                   extent=heatmap_data['extent'],
                   origin='lower',
                   aspect='auto',
                   alpha=0.8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Message Density', rotation=270, labelpad=20)

    # Plot agent positions if requested
    if show_agents:
        positions = heatmap_data['positions']
        actions = heatmap_data['actions']

        # Different markers for different actions
        no_msg = actions == 0
        cam_only = actions == 1
        cpm_only = actions == 2
        both_msg = actions == 3

        # Plot agents with different styles
        if np.any(no_msg):
            ax.scatter(positions[no_msg, 0], positions[no_msg, 1],
                       c='gray', s=20, alpha=0.5, marker='o', label='No message')
        if np.any(cam_only):
            ax.scatter(positions[cam_only, 0], positions[cam_only, 1],
                       c='blue', s=30, alpha=0.7, marker='^', label='CAM only')
        if np.any(cpm_only):
            ax.scatter(positions[cpm_only, 0], positions[cpm_only, 1],
                       c='red', s=30, alpha=0.7, marker='s', label='CPM only')
        if np.any(both_msg):
            ax.scatter(positions[both_msg, 0], positions[both_msg, 1],
                       c='purple', s=40, alpha=0.8, marker='*', label='Both')

        ax.legend(loc='upper right', framealpha=0.9)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_combined_heatmap(heatmap_data: Dict[str, np.ndarray],
                          step: int,
                          save_path: str = None):
    """Plot CAM and CPM heatmaps side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    extent = heatmap_data['extent']

    # CAM heatmap
    im1 = ax1.imshow(heatmap_data['cam'],
                     cmap='Blues',
                     extent=extent,
                     origin='lower',
                     aspect='auto',
                     alpha=0.8)
    ax1.set_title(f'CAM Message Density - Step {step}', fontsize=12)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    plt.colorbar(im1, ax=ax1, label='Density')

    # CPM heatmap
    im2 = ax2.imshow(heatmap_data['cpm'],
                     cmap='Reds',
                     extent=extent,
                     origin='lower',
                     aspect='auto',
                     alpha=0.8)
    ax2.set_title(f'CPM Message Density - Step {step}', fontsize=12)
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    plt.colorbar(im2, ax=ax2, label='Density')

    # Add agent positions
    positions = heatmap_data['positions']
    actions = heatmap_data['actions']

    for ax in [ax1, ax2]:
        # Plot all agents as small dots
        ax.scatter(positions[:, 0], positions[:, 1],
                   c='black', s=5, alpha=0.3, marker='.')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Communication Message Density Analysis - Step {step}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """Main script to generate spatial heatmaps."""
    import json

    # Set matplotlib backend for consistency
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for consistent output

    # Configuration
    model_dir = "mappo_runs/20250619_190642"  # Update with your model directory
    checkpoint = "best_model.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Output directory
    output_dir = "output/spatial_heatmaps"
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    with open(f"{model_dir}/config.json", 'r') as f:
        config = json.load(f)

    alpha = config.get('alpha', 1.0)
    beta = config.get('beta', 0.001)

    # Steps to analyze
    start_step = 1
    end_step = 1500  # Adjust based on your data
    step_interval = 100  # Save heatmap every 100 steps

    # Create environment
    step_list = list(range(start_step, end_step + 1))
    env = ComNetEnv(
        max_agents=config['max_agents'],
        step_list=step_list,
        alpha=alpha,
        beta=beta,
        normalize_reward=True,
        share_reward=True,
        obs_include_global=True
    )

    # Load model
    obs_dim = env.observation_space.shape[0]
    actor = Actor(obs_dim=obs_dim).to(device)

    checkpoint_path = f"{model_dir}/{checkpoint}"
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint_data['actor_state_dict'])
    actor.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Generating heatmaps for steps {start_step} to {end_step}")

    # Storage for GIF frames
    gif_frames = []
    heatmap_stats = []

    # Reset environment
    obs = env.reset()

    # Process each step
    for step_idx, actual_step in enumerate(tqdm(step_list, desc="Processing steps")):
        try:
            # Get actions from policy
            with torch.no_grad():
                active_obs = obs[:env.num_agents]
                obs_tensor = torch.FloatTensor(active_obs).to(device)
                dist = actor(obs_tensor)
                actions = dist.sample().cpu().numpy()

            # Load positions
            positions, entity_ids = load_position_data(actual_step)

            # Ensure we have the right number of positions
            if len(positions) > env.num_agents:
                # Too many positions, truncate
                positions = positions[:env.num_agents]
            elif len(positions) < env.num_agents:
                # Too few positions, pad with random positions around mean
                print(
                    f"Warning: Position count mismatch at step {actual_step} ({len(positions)} positions, {env.num_agents} agents)")
                if len(positions) > 0:
                    mean_pos = positions.mean(axis=0)
                    std_pos = positions.std(axis=0)
                    if np.any(std_pos == 0):
                        std_pos = np.ones(2) * 10
                else:
                    mean_pos = np.zeros(2)
                    std_pos = np.ones(2) * 100

                # Generate additional positions
                n_missing = env.num_agents - len(positions)
                extra_positions = np.random.normal(mean_pos, std_pos, (n_missing, 2))
                positions = np.vstack([positions, extra_positions]) if len(positions) > 0 else extra_positions

            # Create heatmap
            heatmap_data = create_spatial_heatmap(positions, actions, grid_size=100, sigma=3.0)

            # Collect statistics
            cam_count = np.sum(actions == 1) + np.sum(actions == 3)
            cpm_count = np.sum(actions == 2) + np.sum(actions == 3)
            total_messages = np.sum(actions > 0)

            heatmap_stats.append({
                'step': actual_step,
                'cam_messages': int(cam_count),
                'cpm_messages': int(cpm_count),
                'total_messages': int(total_messages),
                'cam_density_max': float(heatmap_data['cam'].max()),
                'cpm_density_max': float(heatmap_data['cpm'].max())
            })

            # Save individual heatmaps every N steps
            if actual_step % step_interval == 0 or actual_step == start_step or actual_step == end_step:
                # Save combined CAM/CPM view
                plot_combined_heatmap(heatmap_data, actual_step,
                                      save_path=f"{output_dir}/combined_step_{actual_step:04d}.png")

                # Save total message density
                plot_heatmap(heatmap_data, actual_step, message_type='total',
                             save_path=f"{output_dir}/total_step_{actual_step:04d}.png",
                             show_agents=True)

            # Create frame for GIF (total message density)
            # Use fixed figure size and DPI to ensure consistent frame dimensions
            fig = plt.figure(figsize=(8, 6), dpi=100)
            ax = fig.add_subplot(111)

            im = ax.imshow(heatmap_data['total'],
                           cmap='viridis',
                           extent=heatmap_data['extent'],
                           origin='lower',
                           aspect='auto')
            ax.set_title(f'Message Density - Step {actual_step}', fontsize=12)
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            plt.colorbar(im, ax=ax, label='Density')

            # Ensure tight layout before saving
            plt.tight_layout()

            # Save frame with explicit bbox to ensure consistent size
            frame_path = f"{output_dir}/frame_{step_idx:04d}.png"
            plt.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            gif_frames.append(frame_path)

        except Exception as e:
            print(f"Error processing step {actual_step}: {e}")
            # Skip this step but continue processing
            continue

        # Step environment
        if step_idx < len(step_list) - 1:
            obs, _, done, _ = env.step(np.zeros(env.max_agents, dtype=int))  # Dummy step
            if done:
                obs = env.reset()

    # Create GIF
    if gif_frames:
        print("\nCreating animated GIF...")
        gif_path = f"{output_dir}/message_density_animation.gif"

        # Read frames and ensure consistent size
        images = []
        target_size = None

        for i, frame_path in enumerate(gif_frames[::5]):  # Use every 5th frame to reduce size
            try:
                img = Image.open(frame_path)

                # Set target size from first image
                if target_size is None:
                    target_size = img.size

                # Resize if necessary
                if img.size != target_size:
                    print(f"Resizing frame from {img.size} to {target_size}")
                    img = img.resize(target_size, Image.Resampling.LANCZOS)

                # Convert to numpy array
                images.append(np.array(img))

            except Exception as e:
                print(f"Warning: Could not read frame {frame_path}: {e}")

        if images:
            try:
                # Save GIF with proper duration (in seconds)
                imageio.mimsave(gif_path, images, duration=200, loop=0)  # duration in milliseconds
                print(f"Saved animated GIF to {gif_path}")
            except Exception as e:
                print(f"Error creating GIF: {e}")
                # Try alternative approach with PIL
                try:
                    print("Trying alternative GIF creation method...")
                    pil_images = [Image.fromarray(img) for img in images]
                    pil_images[0].save(
                        gif_path,
                        save_all=True,
                        append_images=pil_images[1:],
                        duration=200,
                        loop=0
                    )
                    print(f"Successfully saved GIF using alternative method to {gif_path}")
                except Exception as e2:
                    print(f"Alternative method also failed: {e2}")
        else:
            print("Warning: No valid frames for GIF creation")

        # Clean up frame files
        for frame_path in gif_frames:
            try:
                os.remove(frame_path)
            except:
                pass
    else:
        print("Warning: No frames generated for GIF")

    # Save statistics
    if heatmap_stats:
        import pandas as pd
        stats_df = pd.DataFrame(heatmap_stats)
        stats_df.to_csv(f"{output_dir}/heatmap_statistics.csv", index=False)

        # Plot summary statistics
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Message counts over time
        axes[0].plot(stats_df['step'], stats_df['cam_messages'], 'b-', label='CAM', alpha=0.7)
        axes[0].plot(stats_df['step'], stats_df['cpm_messages'], 'r-', label='CPM', alpha=0.7)
        axes[0].plot(stats_df['step'], stats_df['total_messages'], 'g-', label='Total', alpha=0.7)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Number of Messages')
        axes[0].set_title('Message Counts Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Maximum density over time
        axes[1].plot(stats_df['step'], stats_df['cam_density_max'], 'b-', label='CAM Max Density', alpha=0.7)
        axes[1].plot(stats_df['step'], stats_df['cpm_density_max'], 'r-', label='CPM Max Density', alpha=0.7)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Maximum Density')
        axes[1].set_title('Peak Message Density Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/statistics_summary.png", dpi=150)
        plt.close()

        print(f"\nAnalysis complete!")
        print(f"Individual heatmaps saved to: {output_dir}")
        print(f"Statistics saved to: {output_dir}/heatmap_statistics.csv")
        print(f"Summary plots saved to: {output_dir}/statistics_summary.png")
    else:
        print("\nWarning: No statistics collected")
        print(f"Check output directory for any saved heatmaps: {output_dir}")


if __name__ == "__main__":
    main()