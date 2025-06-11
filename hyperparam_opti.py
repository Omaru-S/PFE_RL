#!/usr/bin/env python3
"""
sweep_efficiency.py
===================

Grid / random sweep over the (alpha, beta) reward weights to find an efficient
knowledge‑per‑byte trade‑off.

Modes
-----
1. **Full**   (Default)   – standard MAPPO training (slowest).
2. **Quick**  (`--quick`) – fine‑tune just a few PPO updates (≈ 10× faster).
3. **Proxy**  (`--proxy`) – *no* RL training; replay a naive policy on a random
   subset of steps and score `alpha·knowledge − beta·bytes` (≈ 100× faster).

The script writes a CSV summary and, except in proxy mode, saves a model
checkpoint & evaluation JSON for every (alpha, beta) pair.

Fix v1.1
--------
* **Bug‑fix** JSON serialization error when `eval_stats` contains NumPy scalars.
  All NumPy scalar/int/float values are now converted to built‑in Python types
  before calling `json.dump()`.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import itertools
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import trange

# Project imports -----------------------------------------------------------
from communication_env import ComNetEnv
from policies import naive_policy
from com_simulation import (
    compute_knowledge,
    compute_bytes,
    sum_knowledge,
    sum_bytes,
)

# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

##############################################################################
# Helper: ensure JSON serialisable objects                                    #
##############################################################################

def _to_py(obj):  # noqa: D401 – simple helper
    """Recursively convert NumPy scalars/arrays to native Python objects."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    return obj

##############################################################################
# 1.  Proxy score (no RL)                                                    #
##############################################################################

def _proxy_metrics(alpha: float, beta: float, steps: List[int], sample: int = 200) -> dict:
    """Return knowledge, bytes, efficiency & reward for a (alpha,beta) pair."""
    total_know = total_bytes = 0
    for step in random.sample(steps, min(sample, len(steps))):
        cam, cpm = naive_policy(step)
        know, objs = compute_knowledge(step, cam, cpm)
        bytes_ = compute_bytes(cam, cpm, objs)
        total_know += sum_knowledge(know)
        total_bytes += sum_bytes(bytes_)

    eff = total_know / (total_bytes + 1e-9)
    reward = alpha * total_know - beta * total_bytes
    return {
        "total_knowledge": float(total_know),
        "total_bytes": float(total_bytes),
        "efficiency": float(eff),
        "proxy_reward": float(reward),
    }

##############################################################################
# 2.  Lightweight MAPPO training / fine‑tune                                 #
##############################################################################

def _train_run(
    alpha: float,
    beta: float,
    *,
    quick: bool = False,
    baseline_ckpt: str | None = None,
    train_steps: List[int] | None = None,
    test_steps: List[int] | None = None,
) -> Tuple[str, dict]:
    """Train (or quick‑tune) a MAPPO agent; return model dir & eval stats."""
    from train_mappo import (
        Actor,
        Critic,
        RolloutBuffer,
        collect_rollout,
        ppo_update,
        evaluate,
    )

    # Directories -----------------------------------------------------------
    tag = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    mdl_dir = f"mappo_runs/sweep_{tag}_a{alpha}_b{beta}"
    os.makedirs(mdl_dir, exist_ok=True)

    # Environments ----------------------------------------------------------
    max_agents = 600
    train_steps = train_steps or list(range(1, 1201))
    test_steps = test_steps or list(range(1201, 1501))

    env_cfg = dict(
        max_agents=max_agents,
        normalize_reward=True,
        share_reward=True,
        obs_include_global=True,
    )
    env = ComNetEnv(step_list=train_steps, alpha=alpha, beta=beta, **env_cfg)
    test_env = ComNetEnv(step_list=test_steps, alpha=alpha, beta=beta, **env_cfg)

    # Hyper‑params ----------------------------------------------------------
    roll_len = 64 if quick else 128
    total_ts = 5_000 if quick else 50_000
    updates = total_ts // roll_len
    lr = 3e-4

    # Networks --------------------------------------------------------------
    obs_dim = env.observation_space.shape[0]
    env.reset()
    state_dim = len(env.get_state())
    actor = Actor(obs_dim).to(DEVICE)
    critic = Critic(state_dim).to(DEVICE)

    # Load baseline if given -----------------------------------------------
    if baseline_ckpt and Path(baseline_ckpt).exists():
        ckpt = torch.load(baseline_ckpt, map_location=DEVICE)
        actor.load_state_dict(ckpt["actor_state_dict"])
        critic.load_state_dict(ckpt["critic_state_dict"])
        print(f"Loaded baseline weights from {baseline_ckpt}")

    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr)

    buf = RolloutBuffer(
        max_agents=max_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        rollout_length=roll_len,
    )

    # Training loop ---------------------------------------------------------
    for _ in trange(updates, desc=f"α={alpha},β={beta},quick={quick}", leave=False):
        buf.ptr = 0
        collect_rollout(env, actor, critic, buf, roll_len, DEVICE)
        ppo_update(
            actor,
            critic,
            opt_a,
            opt_c,
            buf.get(),
            clip_ratio=0.2,
            entropy_coef=0.01,
            num_epochs=2 if quick else 4,
            batch_size=64,
            device=DEVICE,
        )

    # Evaluate --------------------------------------------------------------
    eval_stats_raw = evaluate(test_env, actor, num_episodes=3, device=DEVICE)
    eval_stats = _to_py(eval_stats_raw)  # ensure JSON serialisable

    # Save ------------------------------------------------------------------
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "eval": eval_stats,
    }, f"{mdl_dir}/final.pt")

    with open(f"{mdl_dir}/eval.json", "w") as f:
        json.dump(eval_stats, f, indent=2)

    return mdl_dir, eval_stats

##############################################################################
# 3.  Main CLI                                                               #
##############################################################################

def main() -> None:  # noqa: C901 – complexity acceptable for script
    p = argparse.ArgumentParser(description="Sweep (alpha,beta) pairs for efficiency")
    p.add_argument("--alpha", nargs="+", type=float, required=True,
                   help="List of alpha values (knowledge weight)")
    p.add_argument("--beta", nargs="+", type=float, required=True,
                   help="List of beta values (byte penalty)")
    p.add_argument("--random", type=int, default=0,
                   help="Randomly sample this many pairs instead of full grid")
    p.add_argument("--quick", action="store_true",
                   help="Short fine‑tune: few PPO updates")
    p.add_argument("--proxy", action="store_true",
                   help="Skip RL training, use proxy replay score")
    p.add_argument("--baseline", type=str,
                   help="Path to baseline checkpoint for quick mode")
    p.add_argument("--csv", default="sweep_results.csv",
                   help="Where to write aggregated CSV (default: sweep_results.csv)")
    args = p.parse_args()

    # Build grid -----------------------------------------------------------
    grid = list(itertools.product(args.alpha, args.beta))
    if args.random > 0:
        grid = random.sample(grid, min(args.random, len(grid)))

    all_steps = list(range(1, 1501))  # needed for proxy replay
    rows: list[dict] = []

    for alpha, beta in grid:
        if args.proxy:
            m = _proxy_metrics(alpha, beta, all_steps)
            row = {"mode": "proxy", "alpha": alpha, "beta": beta, **m}
            print(f"[PROXY] α={alpha:.3g} β={beta:.3g} eff={m['efficiency']:.4f}")
        else:
            dir_, ev = _train_run(alpha, beta, quick=args.quick, baseline_ckpt=args.baseline)
            eff = ev["mean_knowledge"] / (ev["mean_bytes"] + 1e-9)
            row = {
                "mode": "quick" if args.quick else "full",
                "alpha": alpha,
                "beta": beta,
                "model_dir": dir_,
                "mean_reward": ev["mean_reward"],
                "mean_knowledge": ev["mean_knowledge"],
                "mean_bytes": ev["mean_bytes"],
                "efficiency": eff,
            }
            print(f"[TRAIN] α={alpha:.3g} β={beta:.3g} eff={eff:.4f}")
        rows.append(_to_py(row))

    # Write CSV ------------------------------------------------------------
    fieldnames = sorted({k for r in rows for k in r})
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    print(f"\nSaved results to {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
