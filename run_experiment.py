from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, List, Tuple
from datetime import datetime
import sys  # NEU

import numpy as np
import torch

#from insurance_gym import InsuranceEnv
from insurance_gym_2 import InsuranceEnvV2 as InsuranceEnv
from source.exploration_schedules import fixed_eps_schedule, EZGreedy
from source.DQN import DQNAgent

# run_experiment.py
# Benchmark different epsilon strategies (Fixed, EZ-Greedy) on the InsuranceEnv.
# Parameters such as number of episodes, episode length, delay, repetition length, ε, α, γ, and seed are provided via command line.
# Generates a timestamped results folder containing experiment_summary.csv, run_meta.json,
# profiles_<variant>_<seed>.csv and actions_<variant>_<seed>.csv.

def _print_progress(current: int, total: int, schedule_label: str, seed: int) -> None:  # NEU
    width = 30
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    msg = f"\r[{schedule_label} seed {seed}] |{bar}| {current}/{total}"
    print(msg, end="", flush=True)

def get_git_commit() -> str:
    """Return the current Git commit hash."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"])
        return out.decode().strip()
    except Exception:
        return "unknown"


def to_onehot(state: int, dim: int) -> np.ndarray:
    """Convert discrete state index to one-hot encoded vector."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[state] = 1.0
    return vec


def run_single_experiment(
    schedule_label: str,  # which exploration schedule to use
    eps_fn: Callable[[int, int], float] | None,  # function to compute epsilon, or None for EZ schedule
    seed: int,  # random seed for reproducibility
    n_episodes: int,  # number of episodes to run
    horizon: int,  # maximum number of steps per episode
    # delay_min: int,  # min delay parameter for claims in the environment
    # delay_max: int,  # max delay parameter for claims in the environment
    k_repeat: int,  # number of steps to repeat an exploratory action in EZ-Greedy
    gamma: float,  # discount factor for future rewards
    alpha: float,  # learning rate for the Q-learning agent
    eps: float,  # base epsilon for fixed ε-greedy strategy
) -> Tuple[list[float], int, list, list]:
    """Run one trial and return returns, profiles list, and actions per episode."""
    # initialize environment and agent
    env = InsuranceEnv(horizon=horizon, seed=seed) # parameters for the environment

    agent = DQNAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        hidden_dims=(128,128),
        buffer_capacity=10000,
        batch_size=64,
        gamma=gamma,
        lr=alpha,            # α als Learning Rate
        sync_every=1000,
        seed=seed,
    ) # DQN agent with specified parameters

    # retrieve static profiles
    profile_list = env.profiles

    # setup RNG and EZ-Greedy helper if needed
    rng = np.random.default_rng(seed)
    ez_helper = (
        EZGreedy(base_eps=eps, k=k_repeat, rng=rng) if schedule_label == "EZ" else None
    )

    episode_returns: list[float] = [] # list of returns per episode
    episode_actions: list[list[Tuple[int, int]]] = []  # list per episode of (step_profile_idx, action)

    for ep in range(n_episodes):

        obs, info = env.reset()  # reset environment and get initial state and profile_idx
        done = False
        step_idx = 0
        ep_return = 0.0
        actions_this_episode: list[Tuple[int, int]] = []

        while not done:
            # State-Encoding
            state_vec = to_onehot(obs, env.observation_space.n)

            # Aktion auswählen
            if schedule_label == "EZ":
                q_vals = agent.policy_net(torch.from_numpy(state_vec).unsqueeze(0).to(agent.device))
                action = ez_helper.select_action(q_vals.detach().cpu().numpy()[0])  # type: ignore[arg-type]
            else:
                epsilon = eps_fn(ep, step_idx) if eps_fn is not None else 0.0
                action = agent.select_action(state_vec, epsilon)

            # Umgebungsschritt
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store & Learn
            next_state_vec = to_onehot(next_obs, env.observation_space.n)
            agent.store_transition(state_vec, action, reward, next_state_vec, done)
            agent.optimize()

            # Logging
            profile_idx = info["profile_idx"]
            actions_this_episode.append((profile_idx, action))
            ep_return += (gamma ** step_idx) * reward
            obs = next_obs
            step_idx += 1

        episode_actions.append(actions_this_episode)
        episode_returns.append(ep_return)
        _print_progress(ep + 1, n_episodes, schedule_label, seed)  # NEU

    print()  # NEU: Zeilenumbruch nach der Fortschrittsanzeige
    return episode_returns, profile_list, episode_actions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insurance Underwriting Experiment"
    )
    parser.add_argument(
        "--episodes", "-p", type=int, default=300,
        help="Number of episodes per seed"
    )
    parser.add_argument(
        "--horizon", "-l", type=int, default=1000,
        help="Maximum number of steps per episode"
    )
    parser.add_argument(
        "--delay", "-d", type=int, default=10,
        help="Delay parameter for claims"
    )
    parser.add_argument(
        "--k_repeat", "-k", type=int, default=None,
        help="Repetition length for EZ-Greedy (defaults to delay)"
    )
    parser.add_argument(
        "--eps", "-e", type=float, default=0.1,
        help="Base ε for fixed ε-greedy"
    )
    parser.add_argument(
        "--alpha", "-a", type=float, default=0.1,
        help="Learning rate α"
    )
    parser.add_argument(
        "--gamma", "-g", type=float, default=0.99,
        help="Discount factor γ"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=0,
        help="Starting seed (uses seed, seed+1, … seed+9)"
    )
    parser.add_argument(
        "--scale", "-c", type=int, default=0,
        help="EZ Scaling (0 for no scaling, 1 for scaling)"
    )
    args = parser.parse_args()

    n_episodes = args.episodes
    horizon = args.horizon
    delay = args.delay
    k_repeat = args.k_repeat or delay
    eps = args.eps
    alpha = args.alpha
    gamma = args.gamma
    base_seed = args.seed
    scaling = args.scale
    seeds = list(range(base_seed, base_seed + 10))
    if scaling == 1:
        eps_ez = eps / (k_repeat - eps * (k_repeat - 1)) # scale ε for EZ-Greedy
    else:
        eps_ez = eps

    # create timestamped results directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    results_dir = Path("results") / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "git_commit": get_git_commit(),
        "episodes": n_episodes,
        "horizon": horizon,
        "delay": delay,
        "k_repeat": k_repeat,
        "eps": eps,
        "alpha": alpha,
        "gamma": gamma,
        "seeds": seeds,
    }
    with (results_dir / "run_meta.json").open("w") as mf:
        json.dump(meta, mf, indent=2)

    variants: List[Tuple[str, Callable[[int, int], float] | None]] = [
        ("Fixed", fixed_eps_schedule(eps)),
        ("EZ", None),
    ]

    all_rows: List[Tuple[str, int, int, float]] = []  # variant, seed, episode, return

    for label, eps_fn in variants:
        for seed in seeds:
            eps_used = eps_ez if label == "EZ" else eps
            ep_returns, ep_profile_list, ep_action_list = run_single_experiment(
                label, eps_fn, seed, n_episodes, horizon, k_repeat, gamma, alpha, eps_used
            )

            # save profile list for this run
            profiles_file = results_dir / f"profiles_{label}_{seed}.csv"
            with profiles_file.open("w", newline="") as pf:
                writer = csv.writer(pf)
                writer.writerow(["profile_idx","age","region","risk_score"])
                for idx, p in enumerate(ep_profile_list):
                    writer.writerow([idx, p.age, p.region, p.risk_score])

            # save action list for this run
            actions_file = results_dir / f"actions_{label}_{seed}.csv"
            with actions_file.open("w", newline="") as af:
                writer = csv.writer(af)
                writer.writerow(["episode","step","profile_idx","action"])
                for ep_idx, steps in enumerate(ep_action_list, start=1):
                    for step_idx, (p_idx, act) in enumerate(steps):
                        writer.writerow([ep_idx, step_idx, p_idx, act])

            # summary rows
            for ep_idx, r in enumerate(ep_returns, start=1):
                all_rows.append((label, seed, ep_idx, r))

    out_file = results_dir / "experiment_summary.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant","seed","episode","return"])
        writer.writerows(all_rows)

    print(f"Results saved to → {out_file}")
    print(f"Metadata saved to → {results_dir / 'run_meta.json'}")


if __name__ == "__main__":
    main()
