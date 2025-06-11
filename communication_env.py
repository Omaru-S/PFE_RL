import numpy as np
import gym
from gym import spaces
from typing import Tuple, List, Dict, Optional

# Import your existing functions
from com_simulation import (
    compute_knowledge,
    compute_bytes,
    sum_bytes,
    sum_knowledge, compute_max_knowledge
)
from matrix_utils import (
    nrows_communication_matrix,
    nrows_vision_matrix,
    read_capabilities
)


class ComNetEnv(gym.Env):
    """
    Multi-agent environment for learning communication policies.

    Each agent decides whether to send CAM/CPM messages to maximize
    knowledge spread while minimizing communication cost.
    """

    def __init__(
            self,
            max_steps: int = 100,
            alpha: float = 1.0,
            beta: float = 0.001,
            start_step: int = 1,
            normalize_reward: bool = True,
            share_reward: bool = True,
            obs_include_global: bool = False,
            step_list: Optional[List[int]] = None,  # NEW
            max_agents: int = 600,
    ):
        """
        Initialize the communication network environment.

        Parameters
        ----------
        max_steps : int
            Maximum number of steps per episode
        alpha : float
            Weight for knowledge in reward function
        beta : float
            Weight for byte cost in reward function
        start_step : int
            Starting step index for loading matrix files
        normalize_reward : bool
            Whether to normalize reward components
        share_reward : bool
            Whether all agents share the same reward
        obs_include_global : bool
            Whether to include global statistics in observations
        """
        super().__init__()

        self.max_steps = max_steps
        self.alpha = alpha
        self.beta = beta
        self.start_step = start_step
        self.normalize_reward = normalize_reward
        self.share_reward = share_reward
        self.obs_include_global = obs_include_global

        self.step_list = step_list or list(range(start_step, start_step + max_steps))
        self.max_steps = len(self.step_list)
        self.max_agents = max_agents

        # Get agent counts from the first step
        self.num_cam_agents = nrows_communication_matrix(start_step)  # m
        self.num_cpm_agents = nrows_vision_matrix(start_step)  # p
        self.num_agents = self.num_cam_agents  # Total agents

        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # 0: none, 1: CAM, 2: CPM, 3: both

        # Observation space per agent
        obs_dim = self._get_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        from policies import naive_policy

        is_send_cam, is_send_cpm = naive_policy(self.start_step)
        knowledge, objects = compute_max_knowledge(self.start_step, is_send_cam, is_send_cpm)
        bytes_list = compute_bytes(is_send_cam, is_send_cpm, objects)

        self.max_knowledge_per_agent = sum_knowledge(knowledge) / self.num_agents
        self.max_bytes_per_step = sum_bytes(bytes_list)

        # State tracking
        self.current_step = None
        self.prev_is_send_cam = None
        self.prev_is_send_cpm = None
        self.prev_knowledge = None
        self.prev_objects_in_vision = None
        self.prev_bytes = None
        self.episode_knowledge = None
        self.episode_bytes = None

        # Try to load capabilities for additional agent features
        try:
            self.capabilities = read_capabilities()
        except:
            self.capabilities = None

    def _get_obs_dim(self) -> int:
        """Calculate observation dimension based on settings."""
        base_dim = 7  # Base observations per agent
        if self.obs_include_global:
            base_dim += 4  # Add global statistics
        return base_dim

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns
        -------
        np.ndarray
            Initial observations for all agents, shape (max_agents, obs_dim)
        """
        self.current_step = 0

        # Get step-specific agent counts
        step_idx = self.step_list[self.current_step]
        self.num_cam_agents = nrows_communication_matrix(step_idx)
        self.num_cpm_agents = nrows_vision_matrix(step_idx)
        self.num_agents = self.num_cam_agents

        # Initialize padded state arrays (size = max_agents)
        self.prev_is_send_cam = np.zeros(self.max_agents, dtype=int)
        self.prev_is_send_cpm = np.zeros(self.max_agents, dtype=int)
        self.prev_knowledge = np.zeros(self.max_agents)
        self.prev_bytes = np.zeros(self.max_agents)

        # CPM-specific agents may be fewer
        self.prev_objects_in_vision = np.zeros(self.num_cpm_agents)

        # Episode tracking
        self.episode_knowledge = 0.0
        self.episode_bytes = 0

        return self._get_observations()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        """
        Execute one environment step.

        Parameters
        ----------
        actions : np.ndarray
            Actions for all agents, shape (max_agents,)
            Each action in {0, 1, 2, 3}

        Returns
        -------
        observations : np.ndarray
            New observations for all agents (padded to max_agents)
        rewards : np.ndarray
            Rewards for all agents (padded to max_agents)
        done : bool
            Whether episode is finished
        info : dict
            Additional information
        """
        # Update agent counts for current step
        step_idx = self.step_list[self.current_step]
        self.num_cam_agents = nrows_communication_matrix(step_idx)
        self.num_cpm_agents = nrows_vision_matrix(step_idx)
        self.num_agents = self.num_cam_agents

        # Convert actions to communication decisions
        is_send_cam, is_send_cpm = self._actions_to_decisions(actions)

        # Run simulation - only pass the actual agents, not padded
        step_idx = self.step_list[self.current_step]
        knowledge, objects_in_vision = compute_knowledge(
            step_idx,
            is_send_cam[:self.num_cam_agents],  # Slice to actual CAM agents
            is_send_cpm[:self.num_cam_agents]  # Slice to actual agents (CPM uses same base count)
        )

        # Compute bytes - same slicing needed
        bytes_list = compute_bytes(
            is_send_cam[:self.num_cam_agents],
            is_send_cpm[:self.num_cam_agents],
            objects_in_vision
        )

        # Calculate rewards
        total_knowledge = sum_knowledge(knowledge)
        total_bytes = sum_bytes(bytes_list)

        # Normalize if requested
        if self.normalize_reward:
            norm_knowledge = total_knowledge / (self.max_knowledge_per_agent * self.num_agents)
            norm_bytes = total_bytes / self.max_bytes_per_step
        else:
            norm_knowledge = total_knowledge
            norm_bytes = total_bytes

        # Compute shared or individual reward
        if self.share_reward:
            rewards_real = np.full(self.num_agents, self.alpha * norm_knowledge - self.beta * norm_bytes)
        else:
            rewards_real = self._compute_individual_rewards(
                knowledge, bytes_list, is_send_cam, is_send_cpm
            )

        # Pad rewards to max_agents
        rewards = np.zeros(self.max_agents, dtype=np.float32)
        rewards[:self.num_agents] = rewards_real

        # Update state (padded arrays are already full length)
        self.prev_is_send_cam = is_send_cam
        self.prev_is_send_cpm = is_send_cpm
        self.prev_knowledge[:self.num_agents] = knowledge
        self.prev_bytes[:self.num_agents] = bytes_list[:self.num_agents]
        self.prev_objects_in_vision = np.array(objects_in_vision)

        # Episode tracking
        self.episode_knowledge += total_knowledge
        self.episode_bytes += total_bytes

        # Increment step
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Observations (padded to max_agents)
        observations = self._get_observations()

        # Info dictionary
        info = {
            'total_knowledge': total_knowledge,
            'total_bytes': total_bytes,
            'avg_knowledge': total_knowledge / self.num_agents,
            'episode_knowledge': self.episode_knowledge,
            'episode_bytes': self.episode_bytes,
            'objects_in_vision': objects_in_vision,
            'cam_sent': np.sum(is_send_cam[:self.num_agents]),
            'cpm_sent': np.sum(is_send_cpm[:self.num_agents]),
        }

        return observations, rewards, done, info

    def _actions_to_decisions(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert discrete actions to CAM/CPM send decisions.

        Action mapping:
        0 → send nothing
        1 → send only CAM
        2 → send only CPM  
        3 → send both CAM and CPM
        """
        # Initialize full padded decision vectors
        is_send_cam = np.zeros(self.max_agents, dtype=int)
        is_send_cpm = np.zeros(self.max_agents, dtype=int)

        # Only process real agents
        real_actions = actions[:self.num_agents]

        # CAM decision logic
        is_send_cam[:self.num_agents] = np.where(real_actions >= 1, 1, 0)
        is_send_cam[:self.num_agents] = np.where(real_actions == 2, 0, is_send_cam[:self.num_agents])

        # CPM decision logic for CPM-capable agents only
        is_send_cpm[:self.num_cpm_agents] = np.where(
            real_actions[:self.num_cpm_agents] >= 2, 1, 0
        )

        return is_send_cam, is_send_cpm

    def _get_observations(self) -> np.ndarray:
        """
        Optimized vectorized observation generation.
        """
        n = self.num_agents
        M = self.max_agents
        g = self.obs_include_global
        T = self.current_step / self.max_steps

        # Pre-allocate with zeros
        obs_dim = 7 + (4 if g else 0)
        obs = np.zeros((M, obs_dim), dtype=np.float32)

        # Batch fill all agent observations at once
        if n > 0:
            obs[:n, 0] = self.prev_is_send_cam[:n]
            obs[:n, 1] = self.prev_is_send_cpm[:n]
            obs[:n, 2] = self.prev_knowledge[:n] / self.max_knowledge_per_agent
            obs[:n, 3] = self.prev_bytes[:n] / 1000.0
            obs[:n, 4] = np.arange(n, dtype=np.float32) / n
            obs[:n, 5] = (np.arange(n) < self.num_cpm_agents).astype(np.float32)
            obs[:n, 6] = T

            if g and n > 0:
                # Compute global stats once
                tot_know = self.prev_knowledge[:n].sum()
                tot_bytes = self.prev_bytes[:n].sum()
                norm_know = tot_know / (self.max_knowledge_per_agent * n)
                norm_bytes = tot_bytes / self.max_bytes_per_step
                cam_rate = self.prev_is_send_cam[:n].mean()
                cpm_rate = (self.prev_is_send_cpm[:self.num_cpm_agents].mean()
                            if self.num_cpm_agents > 0 else 0.0)

                # Broadcast to all agents
                obs[:n, 7] = norm_know
                obs[:n, 8] = norm_bytes
                obs[:n, 9] = cam_rate
                obs[:n, 10] = cpm_rate

        return obs

    def _compute_individual_rewards(
            self,
            knowledge: List[float],
            bytes_list: List[int],
            is_send_cam: np.ndarray,
            is_send_cpm: np.ndarray
    ) -> np.ndarray:
        """
        Compute individual rewards for each agent based on their contribution.

        This is an alternative to shared rewards, where each agent gets
        credit for their knowledge gain minus their communication cost.
        """
        rewards = np.zeros(self.num_agents)

        for i in range(self.num_agents):
            # Knowledge contribution (normalized)
            if self.normalize_reward:
                knowledge_contrib = knowledge[i] / self.max_knowledge_per_agent
                bytes_cost = bytes_list[i] / 1000.0  # Simple normalization
            else:
                knowledge_contrib = knowledge[i]
                bytes_cost = bytes_list[i]

            # Individual reward
            rewards[i] = self.alpha * knowledge_contrib - self.beta * bytes_cost

        return rewards

    def get_state(self) -> np.ndarray:
        """
        Optimized global state computation with fixed dimensions.
        """
        # Pre-compute statistics
        if self.num_agents > 0:
            total_knowledge = self.prev_knowledge[:self.num_agents].sum()
            total_bytes = self.prev_bytes[:self.num_agents].sum()

            if self.normalize_reward:
                norm_knowledge = total_knowledge / (self.max_knowledge_per_agent * self.num_agents)
                norm_bytes = total_bytes / self.max_bytes_per_step
            else:
                norm_knowledge = total_knowledge
                norm_bytes = total_bytes
        else:
            norm_knowledge = norm_bytes = 0.0

        mean_objects = (self.prev_objects_in_vision.mean() / 10.0
                        if len(self.prev_objects_in_vision) > 0 else 0.0)

        # Create state vector efficiently
        global_state = np.array([
            self.current_step / self.max_steps,
            norm_knowledge,
            norm_bytes,
            mean_objects
        ], dtype=np.float32)

        # Get ALL observations (including padded) to maintain fixed size
        obs = self._get_observations()  # Full size: (max_agents, obs_dim)

        # Concatenate - this ensures fixed state dimension
        state = np.concatenate([obs.flatten(), global_state])

        return state

    def render(self, mode='human'):
        """Simple text rendering of current state."""
        if mode == 'human':
            print(f"\n=== Step {self.current_step}/{self.max_steps} ===")
            print(f"CAM agents sending: {np.sum(self.prev_is_send_cam)}/{self.num_agents}")
            print(f"CPM agents sending: {np.sum(self.prev_is_send_cpm)}/{self.num_cpm_agents}")
            print(f"Total knowledge: {np.sum(self.prev_knowledge):.2f}")
            print(f"Total bytes: {np.sum(self.prev_bytes)}")
            print(f"Avg objects in vision: {np.mean(self.prev_objects_in_vision):.2f}")

    def close(self):
        """Clean up resources."""
        pass

    def seed(self, seed=None):
        """Set random seed."""
        np.random.seed(seed)
        return [seed]


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = ComNetEnv(
        max_steps=10,
        alpha=1.0,
        beta=0.001,
        normalize_reward=True,
        share_reward=True,
        obs_include_global=True
    )

    # Test environment
    obs = env.reset()
    print(f"Initial observations shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run a few steps with random actions
    for step in range(5):
        actions = np.random.randint(0, 4, size=env.num_agents)
        obs, rewards, done, info = env.step(actions)

        print(f"\nStep {step + 1}:")
        print(f"Rewards: mean={np.mean(rewards):.4f}, std={np.std(rewards):.4f}")
        print(f"Info: {info}")
        env.render()

        if done:
            break