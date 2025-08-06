from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, List, Tuple
from datetime import datetime

import numpy as np

from insurance_gym import InsuranceEnv
from source.exploration_schedules import fixed_eps_schedule, EZGreedy
from source.qlearner import QLearner


# run_experiment.py
# Benchmark different epsilon strategies (Fixed, EZ-Greedy) on the InsuranceEnv.
# Parameters such as number of episodes, episode length, delay, repetition length, ε, α, γ, and seed are provided via command line.
# Generates a timestamped results folder containing experiment_summary.csv and run_meta.json.


def get_git_commit() -> str:
    """Return the current Git commit hash."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"])
        return out.decode().strip()
    except Exception:
        return "unknown"


def run_single_experiment(
    schedule_label: str,  # which exploration schedule to use
    eps_fn: Callable[[int, int], float] | None,  # function to compute epsilon, or None for EZ schedule
    seed: int,  # random seed for reproducibility
    n_episodes: int,  # number of episodes to run
    horizon: int,  # maximum number of steps per episode
    delay: int,  # delay parameter for claims in the environment
    k_repeat: int,  # number of steps to repeat an exploratory action in EZ-Greedy
    gamma: float,  # discount factor for future rewards
    alpha: float,  # learning rate for the Q-learning agent
    eps: float,  # base epsilon for fixed ε-greedy strategy
) -> list[float]:
    """Run one trial and return a list of discounted returns, one per episode."""
    # initialize environment and agent
    env = InsuranceEnv(delay=delay, horizon=horizon, seed=seed)
    agent = QLearner(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        seed=seed,
    )

    # setup RNG and EZ-Greedy helper if needed
    rng = np.random.default_rng(seed)
    ez_helper = (
        EZGreedy(base_eps=eps, k=k_repeat, rng=rng) if schedule_label == "EZ" else None
    )

    episode_returns: list[float] = []
  # count of accepted profiles
    for ep in range(n_episodes):
        obs, _ = env.reset()  # reset the environment and get initial observation
        print(obs)  # debug: print initial observation
        done = False
        step_idx = 0
        ep_return = 0.0
        accepts = 0

        while not done:
            if schedule_label == "EZ":
                action = ez_helper.select_action(agent.q[obs])  # type: ignore[arg-type]
            else:
                epsilon = eps_fn(ep, 0) if eps_fn is not None else 0.0
                action = agent.act(obs, epsilon)
            if action == 1:  
                accepts += 1
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, next_obs)
            obs = next_obs

            ep_return += (gamma ** step_idx) * reward
            step_idx += 1
            done = terminated or truncated

        episode_returns.append(ep_return)

    return episode_returns, accepts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insurance Underwriting Experiment"
    )
    parser.add_argument(
        "--episodes", "-e", type=int, default=300,
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
        "--eps", "-ε", type=float, default=0.1,
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
    args = parser.parse_args()

    n_episodes = args.episodes
    horizon = args.horizon
    delay = args.delay
    k_repeat = args.k_repeat or delay
    eps = args.eps
    alpha = args.alpha
    gamma = args.gamma
    base_seed = args.seed
    seeds = list(range(base_seed, base_seed + 10))

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

    all_rows: List[Tuple[str, int, int, float, int]] = []  # variant, seed, episode, return, accepts

    for label, eps_fn in variants:
        for seed in seeds:
            ep_returns, ep_accepts = run_single_experiment(
                label, eps_fn, seed, n_episodes, horizon, delay, k_repeat, gamma, alpha, eps
            )
            for ep_idx, r in enumerate(ep_returns, start=1):
                all_rows.append((label, seed, ep_idx, r, ep_accepts))

    out_file = results_dir / "experiment_summary.csv"
    with out_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant","seed","episode","return", "accepts"])
        writer.writerows(all_rows)

    print(f"Results saved to → {out_file}")
    print(f"Metadata saved to → {results_dir / 'run_meta.json'}")


if __name__ == "__main__":
    main()
