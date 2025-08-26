#!/usr/bin/env python3
"""
test_policy_generalization.py
================================

Evaluate a trained multi‑agent communication policy **and** compare it
against the built‑in *naive* baseline under a family of out‑of‑distribution
perturbations:

* **Density sweep** ρ ∈ {0, 0.2, 0.4, 0.6, 0.8} — scales every link/vision
  success‑probability by (1 − ρ).
* **Sensor noise**  σ ∈ {0.01, 0.05, 0.10} — i.i.d. Gaussian noise added to
  each observation element.

For each scenario the script logs

* mean episode reward
* reward variance
* success‑rate (positive total reward)
* average timesteps to termination

and stores **three bar‑charts** (reward, knowledge, bytes) comparing
*trained* vs. *naive* policies, exactly like those produced by
`analyze_mappo`.

Outputs
-------
* Pretty console table
* `generalization_results.csv`
* `generalization_results.png` (aggregate mean rewards)
* `comparison_figs/{scenario}_{metric}_compare.png` trio for **every**
  scenario

Dependencies
------------
`numpy`, `pandas`, `matplotlib`, `torch`, `gym`, and the project modules
already present in *PFE_RLv2*.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import gym

# Project‑specific imports -------------------------------------------------
from communication_env import ComNetEnv
import com_simulation as cs
from matrix_utils import read_communication_matrix, read_vision_matrix
from policies import naive_policy  # reference baseline provided by repo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------------
# Observation‑noise wrapper (API‑agnostic)
# -------------------------------------------------------------------------
class NoisyObservationWrapper(gym.Wrapper):
    """Adds Gaussian N(0, σ²) noise to observations regardless of Gym API."""

    def __init__(self, env: gym.Env, sigma: float):
        super().__init__(env)
        self.sigma = float(sigma)

    # Helper --------------------------------------------------------------
    def _add_noise(self, obs):
        if self.sigma > 0:
            obs = obs + np.random.normal(0.0, self.sigma, obs.shape).astype(obs.dtype)
        return obs

    # Modern/legacy reset --------------------------------------------------
    def reset(self, **kwargs):  # pyright: ignore[override]
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs = self._add_noise(out[0])
            return (obs, *out[1:])
        return self._add_noise(out)

    # Modern/legacy step ---------------------------------------------------
    def step(self, action):  # pyright: ignore[override]
        out = self.env.step(action)
        if isinstance(out, tuple):
            obs = self._add_noise(out[0])
            return (obs, *out[1:])
        return self._add_noise(out)

# -------------------------------------------------------------------------
# Density scaling helpers
# -------------------------------------------------------------------------

def _make_density_steps(rho: float):
    """Return patched Bernoulli functions with success‑probability scaled."""
    scale = max(0.0, min(1.0, 1.0 - rho))

    def com_step(step: int):
        p = read_communication_matrix(step) * scale
        return (np.random.random(p.shape) < p).astype(np.uint8)

    def vis_step(step: int):
        p = read_vision_matrix(step) * scale
        return (np.random.random(p.shape) < p).astype(np.uint8)

    return com_step, vis_step


@contextlib.contextmanager
def density_context(rho: float):
    """Temporarily patch com/vision Bernoulli samplers with scaled versions."""
    if not rho:
        yield
        return

    orig_com, orig_vis = cs.com_bernoulli_step, cs.vision_bernoulli_step
    cs.com_bernoulli_step, cs.vision_bernoulli_step = _make_density_steps(rho)
    try:
        yield
    finally:
        cs.com_bernoulli_step, cs.vision_bernoulli_step = orig_com, orig_vis

# -------------------------------------------------------------------------
# Policy loading
# -------------------------------------------------------------------------
class TorchPolicyWrapper:
    """Wrap a Torch actor so it exposes `.predict(obs)->actions`."""

    def __init__(self, actor: torch.nn.Module, device: str = DEVICE):
        self.actor, self.device = actor.eval(), device

    def predict(self, obs: np.ndarray) -> np.ndarray:  # noqa: D401
        with torch.no_grad():
            tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            dist = self.actor(tensor)
            return dist.sample().cpu().numpy()


def _load_policy_from_pt(pt_path: str, env: ComNetEnv):
    from train_mappo import Actor  # noqa: F401 – implicit shape inference
    from analyze_mappo import load_model as _lm

    obs_dim = env.observation_space.shape[0]
    env.reset()
    state_dim = len(env.get_state())

    actor, _, _ = _lm(pt_path, obs_dim, state_dim, DEVICE)
    return TorchPolicyWrapper(actor, DEVICE)


def load_policy(policy_path: str, env: ComNetEnv):
    """Return object exposing `.predict`. Supports .pt/.pth or Python module."""
    policy_path = os.path.expanduser(policy_path)

    if policy_path.endswith((".pt", ".pth")):
        return _load_policy_from_pt(policy_path, env)

    spec = importlib.util.spec_from_file_location("user_policy", policy_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)  # type: ModuleType
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        if hasattr(module, "policy"):
            return module.policy
        if hasattr(module, "Policy"):
            return module.Policy()
        raise AttributeError("Expected `policy` instance or `Policy` class in" f" {policy_path}")

    raise FileNotFoundError(policy_path)

# -------------------------------------------------------------------------
# Env builder & evaluation loops
# -------------------------------------------------------------------------

def make_env_variant(*, rho: float = 0.0, sigma: float = 0.0, seed: int | None = None) -> ComNetEnv:
    env = ComNetEnv(
        max_agents=600,
        normalize_reward=True,
        share_reward=True,
        obs_include_global=True,
    )
    if seed is not None:
        env.seed(seed)
    if sigma > 0:
        env = NoisyObservationWrapper(env, sigma)
    return env


def evaluate_policy(env: ComNetEnv, policy, episodes: int = 10) -> Dict[str, float]:
    rewards, steps, successes = [], [], 0
    for _ in range(episodes):
        obs = env.reset()
        done, ep_reward, step_count = False, 0.0, 0
        while not done:
            n_act = env.num_agents
            act = policy.predict(obs[:n_act])
            full = np.zeros(env.max_agents, dtype=int)
            full[:n_act] = act
            obs, r, done, _ = env.step(full)
            ep_reward += r[:n_act].sum()
            step_count += 1
        rewards.append(ep_reward)
        steps.append(step_count)
        successes += int(ep_reward > 0)

    r_np = np.asarray(rewards, dtype=np.float32)
    return {
        "mean_reward": r_np.mean(),
        "reward_var": r_np.var(ddof=0),
        "success_rate": successes / episodes,
        "avg_steps": np.mean(steps),
    }

# -------------------------------------------------------------------------
# Trained vs Naive comparison
# -------------------------------------------------------------------------

# ──────────────────────────────────────────────────────────────────────────────
#  PATCH – updated plotting for visual continuity with analyse_mappo.py
#  Only the helper _plot_metric() and evaluate_vs_naive() are shown; the rest
#  of the original script stays exactly the same.
#  Drop these definitions straight into test_policy_generalization.py, **replacing**
#  the current evaluate_vs_naive() implementation.
# ──────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# … (other imports remain unchanged)

# -------------------------------------------------------------------------
# Helper: consistent bar‑plot styling (matches analyse_mappo.compare_policies)
# -------------------------------------------------------------------------

def _plot_metric(ax, metric_key: str, trained: dict, naive: dict, title: str):
    """Draw bar‑chart with error‑bars & ±% annotation – identical style to
    analyse_mappo.compare_policies().
    """
    trained_mean = trained[metric_key].mean()
    trained_std = trained[metric_key].std(ddof=0)
    naive_mean = naive[metric_key].mean()
    naive_std = naive[metric_key].std(ddof=0)

    # Order & colours identical across scripts ⇒ easier side‑by‑side reading
    x = ["Naive", "Trained"]
    y = [naive_mean, trained_mean]
    err = [naive_std, trained_std]

    bars = ax.bar(x, y, yerr=err, capsize=10)

    # Title & axis labels --------------------------------------------------
    ax.set_title(title)
    ax.set_ylabel("Value")

    # Improvement / reduction annotation ----------------------------------
    if metric_key == "bytes":
        pct = (naive_mean - trained_mean) / (naive_mean + 1e-9) * 100.0
        sign = "reduction"
        colour = "green"
    else:
        pct = (trained_mean - naive_mean) / (naive_mean + 1e-9) * 100.0
        sign = "improvement"
        colour = "green" if pct >= 0 else "red"

    top = max(y) + max(err) * 1.2
    ax.text(0.5, top, f"{pct:+.1f}% {sign}", ha="center", fontsize=10, color=colour)

    # Clean aesthetics -----------------------------------------------------
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# -------------------------------------------------------------------------
#  MAIN COMPARISON PLOTTER (replaces old evaluate_vs_naive)
# -------------------------------------------------------------------------

def evaluate_vs_naive(env: "ComNetEnv",
                      trained_policy,
                      episodes: int = 10,
                      out_dir: str | None = None,
                      tag: str = "") -> dict[str, dict[str, np.ndarray]]:
    """Roll out *trained* vs *naive* and save three comparison bar‑charts with the
    exact same visual style as analyse_mappo.compare_policies().
    Returns the per‑episode arrays as before.
    """
    from policies import naive_policy  # local import to avoid circularity

    # -----------------------------
    # Roll‑out helpers (unchanged)
    # -----------------------------
    def rollout(policy_name: str):
        R, K, B = [], [], []
        for _ in range(episodes):
            obs, done = env.reset(), False
            ep_r = ep_k = ep_b = 0.0
            while not done:
                if policy_name == "trained":
                    act = trained_policy.predict(obs[:env.num_agents])
                    actions = np.zeros(env.max_agents, dtype=int)
                    actions[:env.num_agents] = act
                else:  # naive baseline
                    step_idx = env.step_list[env.current_step]
                    send_cam, send_cpm = naive_policy(step_idx)
                    actions = np.ones(env.max_agents, dtype=int)  # default CAM
                    actions[:env.num_cpm_agents] = np.where(send_cpm[:env.num_cpm_agents], 3, 1)

                obs, rew, done, info = env.step(actions)
                ep_r += rew[:env.num_agents].sum()
                ep_k += info.get("total_knowledge", 0.0)
                ep_b += info.get("total_bytes", 0.0)
            R.append(ep_r); K.append(ep_k); B.append(ep_b)
        return {
            "rewards": np.asarray(R, dtype=np.float32),
            "knowledge": np.asarray(K, dtype=np.float32),
            "bytes": np.asarray(B, dtype=np.float32),
        }

    trained = rollout("trained")
    naive = rollout("naive")

    # -----------------------------
    # Plots (new consistent style)
    # -----------------------------
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        metrics_titles = [
            ("rewards", "Total Reward"),
            ("knowledge", "Total Knowledge"),
            ("bytes", "Total Bytes"),
        ]

        for metric_key, title in metrics_titles:
            fig, ax = plt.subplots(figsize=(6, 4))  # same size as analyse_mappo
            _plot_metric(ax, metric_key, trained, naive, title)
            fig.tight_layout()
            fname = Path(out_dir) / f"{tag}_{metric_key}_compare.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)

    return {"trained": trained, "naive": naive}

# ──────────────────────────────────────────────────────────────────────────────
# NOTE: everything *below* this line is exactly the original content of
#       test_policy_generalization.py and has been omitted here for brevity.
#       Keep the rest of the script unchanged.
# ──────────────────────────────────────────────────────────────────────────────


# -------------------------------------------------------------------------
# Results logging
# -------------------------------------------------------------------------

def log_results(rows: List[Dict],
                csv_path: str = "generalization_results.csv",
                fig_path: str = "generalization_results.png") -> None:
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False, float_format="%.4f"))

    df.to_csv(csv_path, index=False)
    print(f"\n[✓] Results saved to {csv_path}")

    plt.figure(figsize=(10, 4))
    plt.bar(df["scenario"], df["mean_reward"])
    plt.ylabel("Mean episode reward")
    plt.title("Policy generalisation – mean reward per scenario")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[✓] Plot saved to {fig_path}")

# -------------------------------------------------------------------------
# Main orchestration
# -------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate policy robustness & compare to naive baseline.")
    ap.add_argument("--policy-path", type=str, required=True,
                    help="Path to .pt checkpoint or Python file with a `.predict` policy.")
    ap.add_argument("--episodes", type=int, default=10, help="Episodes per scenario (default: 10)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = ap.parse_args()

    np.random.seed(args.seed)

    # Scenarios ----------------------------------------------------------
    density_levels = np.linspace(0.0, 0.8, 5)
    noise_sigmas = [0.01, 0.05, 0.10]
    scenarios: List[Dict] = (
        [{"scenario": "baseline", "rho": 0.0, "sigma": 0.0}] +
        [{"scenario": f"density_{rho:.2f}", "rho": float(rho), "sigma": 0.0} for rho in density_levels] +
        [{"scenario": f"noise_{sig:.2f}", "rho": 0.0, "sigma": float(sig)} for sig in noise_sigmas]
    )

    # Load policy --------------------------------------------------------
    ref_env = make_env_variant(seed=args.seed)
    policy = load_policy(args.policy_path, ref_env)
    ref_env.close()

    results = []
    for sc in scenarios:
        env = make_env_variant(rho=sc["rho"], sigma=sc["sigma"], seed=args.seed)
        with density_context(sc["rho"]):
            metrics = evaluate_policy(env, policy, args.episodes)
            cmp = evaluate_vs_naive(env, policy, args.episodes,
                                    out_dir="comparison_figs", tag=sc["scenario"])
        env.close()

        results.append({
            "scenario": sc["scenario"],
            **metrics,
            "naive_reward": cmp["naive"]["rewards"].mean(),
            "rho": sc["rho"],
            "sigma": sc["sigma"],
        })

        print(f"[{sc['scenario']}]  mean_reward={metrics['mean_reward']:.2f}  "
              f"(+{metrics['mean_reward'] - cmp['naive']['rewards'].mean():.1f} vs naïve)")

    log_results(results)


if __name__ == "__main__":
    main()
