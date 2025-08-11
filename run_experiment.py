from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Callable, List, Tuple
from datetime import datetime

import numpy as np
import torch

from insurance_gym_2 import InsuranceEnvV2 as InsuranceEnv
from source.exploration_schedules import EZGreedy, fixed_eps_schedule, annealed_linear
from source.DQN import DQNAgent



def _print_progress(current: int, total: int, schedule_label: str, seed: int) -> None:
    """
    Print a progress bar to the console
    """
    width = 30
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    msg = f"\r[{schedule_label} seed {seed}] |{bar}| {current}/{total}"
    print(msg, end="", flush=True)

def get_git_commit() -> str:
    """
    Return the current Git commit hash.
    """
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"])
        return out.decode().strip()
    except Exception:
        return "unknown"

def evaluate_policy(env, # DQNAgent,
                    agent, # InsuranceEnv,
                    n_episodes: int, # number of episodes to evaluate,
                    max_steps: int, # maximum number of steps per episode,
                    base_seed: int, # base seed for reproducibility,
                    block_ep: int, # current block episode number,
                    label: str, # label for the evaluation variant,
                    seed: int): #   random seed for reproducibility
    
    """
    Evaluate a trained policy in a greedy (ε=0) setting over multiple episodes.

    This function runs the given agent in the provided environment without exploration, 
    selecting the action with the highest predicted Q-value at each step. 
    It collects key performance metrics per episode, such as total return, bankruptcy occurrence, 
    time to ruin, capital trajectory, liabilities, and terminal payouts.
    """

    rows = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=base_seed + i)
        ep_return = 0.0 # total return for this episode
        min_capital = float(info.get("capital", 0.0)) # minimum capital during the episode
        time_to_ruin = None # time step when bankruptcy occurred, if any
        final_capital = min_capital # final capital at the end of the episode
        liabilities_end = 0.0 # total liabilities at the end of the episode
        terminal_paid = 0.0 # total terminal payouts at the end of the episode
        for t in range(max_steps):
            s = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(agent.device) # add batch dimension
            with torch.no_grad():
                q = agent.policy_net(s) # compute Q-values
            action = int(q.argmax(dim=1).item()) # select action with highest Q-value
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward) # accumulate discounted return
            cap = float(info.get("capital", min_capital)) # current capital
            min_capital = min(min_capital, cap)
            final_capital = cap
            liabilities_end = float(info.get("liabilities", liabilities_end)) # total liabilities at the end of the episode
            terminal_paid = float(info.get("terminal_paid", terminal_paid)) # total terminal payouts at the end of the episode
            if terminated or truncated:
                bankrupt = 1 if cap < 0.0 else 0
                if bankrupt and time_to_ruin is None:
                    time_to_ruin = t
                break
        rows.append({
            "block_episode": block_ep, # block episode index for logging
            "variant": label, # label for the evaluation variant
            "seed": seed, # evaluation seed
            "episode": i, # episode index
            "return": ep_return, # total accumulated reward
            "bankrupt": int(final_capital < 0.0), # 1 if final capital < 0, else 0
            "time_to_ruin": -1 if time_to_ruin is None else int(time_to_ruin), # step index when bankruptcy occurred, or -1 if none
            "min_capital": min_capital, # minimum capital observed during the episode
            "final_capital": final_capital, # capital at the end of the episode
            "liabilities_end": liabilities_end, # liabilities at the end of the episode
            "terminal_paid": terminal_paid, # terminal payouts made at the end of the episode
        })
    return rows


def run_single_experiment(
    schedule_label: str,  # which exploration schedule to use
    eps_fn: Callable[[int, int], float] | None,  # function to compute epsilon, or None for EZ schedule
    seed: int,  # random seed for reproducibility
    n_episodes: int,  # number of episodes to run
    horizon: int,  # maximum number of steps per episode
    k_repeat: int,  # number of steps to repeat an exploratory action in EZ-Greedy
    gamma: float,  # discount factor for future rewards
    alpha: float,  # learning rate for the Q-learning agent
    eps: float,  # base epsilon for fixed ε-greedy strategy
    min_delay: int,  # minimum delay for claim payouts
    max_delay: int,  # maximum delay for claim payouts
    hidden_dims: Tuple[int, ...], # hidden layer sizes for the DQN       
    batch_size: int, # batch size for DQN updates 
    sync_every: int, # target network sync interval in environment steps                    
    bankruptcy_penalty: float,   # extra penalty applied on bankruptcy      
    buffer_capacity: int, # replay buffer capacity (number of transitions)
    eval_every: int, # run an evaluation block every N training episodes
    eval_episodes: int, # number of eval episodes per evaluation block
    results_dir
) -> Tuple[list[float], list, list, list]:
    
    """
    Run a complete reinforcement learning experiment with a specified exploration strategy.

    This function sets up the insurance underwriting environment, initializes a DQN-based agent, 
    and executes a training loop for the given number of episodes. 
    Depending on the configuration, it can use standard ε-greedy exploration or 
    EZ-Greedy (temporally extended ε-greedy) with optional rate correction. 
    Throughout training, it logs performance metrics, evaluates the agent periodically, and returns all collected results.
    """
    
    # initialize environment and agent
    env = InsuranceEnv(
        horizon=horizon,
        seed=seed,
        delay_min=min_delay,
        delay_max=max_delay,
        bankruptcy_penalty=bankruptcy_penalty,
    )

    agent = DQNAgent(
        state_size=env.observation_space.shape[0], # state size from the environment
        action_size=env.action_space.n, # action size from the environment
        hidden_dims=hidden_dims,  # use specified hidden layer sizes
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        gamma=gamma, # discount factor
        lr=alpha,            # α als Learning Rate
        sync_every=sync_every, # target network sync interval
        seed=seed, # random seed for reproducibility
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
    eval_rows: list[dict] = [] # evaluation results per episode
    train_rows: list[dict] = [] # training results per episode
    for ep in range(n_episodes):

        obs, info = env.reset()  # reset environment and get initial state and profile_idx
        final_capital = float(info.get("capital", 0.0))
        liabilities_end = 0.0 # total liabilities at the end of the episode
        terminal_paid = 0.0 # total terminal payouts at the end of the episode
        min_capital = float(info.get("capital", 0.0))
        bankrupt = 0
        time_to_ruin = None
        exploration_steps = 0
        accept_count = 0
        premium_sum = 0.0
        loss_paid_sum = 0.0
        done = False
        step_idx = 0 
        ep_return = 0.0
        actions_this_episode: list[Tuple[int, int]] = []
        ez_steps = 0 #  total EZ steps in this episode
        ez_phases = 0 # total EZ phases in this episode
        ez_repeats = 0  # total EZ repeats in this episode


        while not done:
            # State-Encoding
            state_vec = obs.astype(np.float32) # convert observation to float32 for the agent
            in_repeat_pre = False # previous action in EZ-Greedy
            argmax_idx = None # action with highest Q-value before exploration

            if schedule_label == "EZ":
                in_repeat_pre = bool(getattr(ez_helper, "steps_left", 0) > 0) # check if we are in a repeat phase
                q_vals = agent.policy_net(torch.from_numpy(state_vec).unsqueeze(0).to(agent.device)) # compute Q-values
                q_np = q_vals.detach().cpu().numpy()[0] # convert to numpy array
                argmax_idx = int(np.argmax(q_np)) # action with highest Q-value
                action = ez_helper.select_action(q_np) # select action using EZ-Greedy
            else:
                epsilon = eps_fn(ep, step_idx) if eps_fn is not None else 0.0 # compute epsilon for the current episode and step
                action, exploration = agent.select_action(state_vec, epsilon)
                if exploration:
                    exploration_steps += 1


            if schedule_label == "EZ":
                steps_left_post = getattr(ez_helper, "steps_left", 0)
                if in_repeat_pre and steps_left_post >= 0:
                    ez_steps += 1
                    ez_repeats += 1
                else:
                    if k_repeat > 1 and steps_left_post == (k_repeat - 1):
                        ez_steps += 1
                        ez_phases += 1
                    elif k_repeat == 1 and int(action) != argmax_idx:
                        ez_steps += 1
                        ez_phases += 1

            #  Step in the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update capital and check for bankruptcy
            final_capital = float(info.get("capital", final_capital))
            liabilities_end = float(info.get("liabilities", liabilities_end))
            terminal_paid = float(info.get("terminal_paid", terminal_paid))
            capital_now = float(info.get("capital", min_capital))
            min_capital = min(min_capital, capital_now)

            # Acceptance / Pricing
            if int(action) > 0:
                accept_count += 1
                if "premium" in info:
                    premium_sum += float(info["premium"])

            # Losses and claims
            if "paid_now" in info:
                loss_paid_sum += float(info["paid_now"])


            # Ruin check
            if (terminated or truncated) and (capital_now < 0.0):
                bankrupt = 1
                if time_to_ruin is None:
                    time_to_ruin = step_idx
            done = terminated or truncated

            # Store & Learn
            next_state_vec = next_obs.astype(np.float32) # convert next observation to float32
            agent.store_transition(state_vec, action, reward, next_state_vec, done) #   store transition in replay buffer
            agent.optimize() # optimize the agent using the replay buffer

            # Logging
            profile_idx = info["profile_idx"]
            actions_this_episode.append((profile_idx, action))
            ep_return += reward
            obs = next_obs
            step_idx += 1

        # Finalize episode
        accept_rate = accept_count / max(1, step_idx)
        avg_premium = (premium_sum / max(1, accept_count)) if accept_count > 0 else 0.0

        episode_actions.append(actions_this_episode)
        episode_returns.append(ep_return)
        train_rows.append({
            "variant": schedule_label,
            "seed": seed,
            "episode": ep + 1,
            "return": float(ep_return),
            "steps": step_idx,
            "accept_rate": float(accept_rate),
            "avg_premium": float(avg_premium),
            "loss_paid_sum": float(loss_paid_sum),
            "min_capital": float(min_capital),
            "final_capital": float(final_capital),
            "bankrupt": int(final_capital < 0.0 or bankrupt == 1),
            "time_to_ruin": -1 if time_to_ruin is None else int(time_to_ruin),
            "ez_steps": int(ez_steps),
            "ez_phases": int(ez_phases),
            "ez_repeats": int(ez_repeats),
            "exploration_steps": int(exploration_steps),
            "exploration_rate": float(exploration_steps) / max(1, step_idx),
        })
        if (ep + 1) % max(1, eval_every) == 0 and eval_episodes > 0:
            eval_rows.extend(
                evaluate_policy(
                    env, agent,
                    n_episodes=eval_episodes,
                    max_steps=horizon,
                    base_seed=seed * 10_000 + (ep + 1) * 100, # unique seed for each eval block
                    block_ep=ep + 1,
                    label=schedule_label,
                    seed=seed,
                )
            )
        _print_progress(ep + 1, n_episodes, schedule_label, seed)

    agent.save_model(
        save_dir=results_dir,
        filename=f"{schedule_label}_seed{seed}.pth"
    )

    print() # Newline after progress bar

    return episode_returns, profile_list, episode_actions, eval_rows, train_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Insurance Underwriting Experiment"
    )
    parser.add_argument(
        "--episodes", "-p", type=int, default=500,
        help="Number of episodes per seed"
    )
    parser.add_argument(
        "--horizon", "-l", type=int, default=500,
        help="Maximum number of steps per episode"
    )
    parser.add_argument(
        "--delay_min", "-dmi", type=int, default=8,
        help="Min Delay parameter for claims"
    )

    parser.add_argument(
        "--delay_max", "-dma", type=int, default=25,
        help="Max Delay parameter for claims"
    )

    parser.add_argument(
        "--k_repeat", "-k", type=int, default=4,
        help="Repetition length for EZ-Greedy (defaults to delay)"
    )
    parser.add_argument(
        "--eps", "-e", type=float, default=0.05,
        help="Base ε for fixed ε-greedy"
    )
    parser.add_argument(
        "--alpha", "-a", type=float, default=0.0004,
        help="Learning rate α"
    )
    parser.add_argument(
        "--gamma", "-g", type=float, default=0.999,
        help="Discount factor γ"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=10,
        help="Starting seed (uses seed, seed+1, … seed+9)"
    )
    parser.add_argument(
        "--scale", "-c", type=int, default=1,
        help="EZ Scaling (0 for no scaling, 1 for scaling)"
    )
    parser.add_argument(
        "--bankruptcy_penalty", "-bp", type=float, default=300_000.0,
        help="Extra penalty applied on bankruptcy"
    )
    parser.add_argument(
        "--hidden_dims", "-hd", type=str, default="256,256",
        help="Comma-separated hidden layer sizes for the DQN, e.g. '128,128' or '256,256,128'"
    )
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=128,
        help="Batch size for DQN updates"
    )
    parser.add_argument(
        "--sync_every", "-se", type=int, default=750,
        help="Target network sync interval in environment steps"
    )
    parser.add_argument(
        "--buffer_capacity", "-bc", type=int, default=200_000,
        help="Replay buffer capacity (number of transitions)"
    )
    parser.add_argument(
        "--eps_start", "-es", type=float, default=1.0,
        help="Starting ε for the annealed schedule"
    )
    parser.add_argument(
        "--eps_end", "-ee", type=float, default=0.05,
        help="Final ε for the annealed schedule"
    )
    parser.add_argument(
        "--eval_every", type=int, default=50,
        help="Run an evaluation block every N training episodes"
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=50,
        help="Number of eval episodes per evaluation block"
    )
    args = parser.parse_args()


    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    buffer_capacity = args.buffer_capacity 
    batch_size = args.batch_size 
    sync_every = args.sync_every  
    bankruptcy_penalty = args.bankruptcy_penalty 
    n_episodes = args.episodes
    horizon = args.horizon
    min_delay = args.delay_min
    max_delay = args.delay_max
    k_repeat = args.k_repeat 
    eps = args.eps
    alpha = args.alpha
    gamma = args.gamma
    base_seed = args.seed
    scaling = args.scale
    eps_start = args.eps_start
    eps_end = args.eps_end
    eval_every=args.eval_every
    eval_episodes=args.eval_episodes

    seeds = list(range(base_seed, base_seed + 10))
    if scaling == 1:
        eps_ez = eps / (k_repeat - eps * (k_repeat - 1)) # scale ε for EZ-Greedy
    else:
        eps_ez = eps

    # create timestamped results directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    results_dir = Path("results") / f"{timestamp}_({n_episodes}ep_{horizon}h_{min_delay}-{max_delay}d_{k_repeat}k_{eps:.2f}e_{alpha:.2f}a_{gamma:.2f}g)"

    results_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "git_commit": get_git_commit(),
        "episodes": n_episodes,
        "horizon": horizon,
        "min_delay": min_delay,
        "max_delay": max_delay,
        "k_repeat": k_repeat,
        "eps": eps,
        "alpha": alpha,
        "gamma": gamma,
        "seeds": seeds,
        "scaling": scaling,
    }
    with (results_dir / "run_meta.json").open("w") as mf:
        json.dump(meta, mf, indent=2)

    variants: List[Tuple[str, Callable[[int, int], float] | None]] = [
        ("Fixed", fixed_eps_schedule(eps)),
        ("EZ", None),
        ("Annealed", annealed_linear(eps_start, eps_end, horizon, n_episodes)),
    ]


    all_rows: List[Tuple[str, int, int, float]] = []  # variant, seed, episode, return

    for label, eps_fn in variants:
        for seed in seeds:
            eps_used = eps_ez if label == "EZ" else eps
            ep_returns, ep_profile_list, ep_action_list, eval_rows, train_rows = run_single_experiment(
                label, eps_fn, seed, n_episodes, horizon, k_repeat, gamma, alpha,
                eps_used, min_delay, max_delay,
                hidden_dims=hidden_dims,
                batch_size=batch_size,
                sync_every=sync_every,
                bankruptcy_penalty=bankruptcy_penalty,
                buffer_capacity=buffer_capacity,
                eval_every=eval_every,
                eval_episodes=eval_episodes,
                results_dir=results_dir
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

            # save training metrics (per episode)
            train_file = results_dir / f"train_metrics_{label}_{seed}.csv"
            with train_file.open("w", newline="") as tf:
                fieldnames = [
                    "variant","seed","episode","return","steps",
                    "accept_rate","avg_premium","loss_paid_sum",
                    "min_capital","final_capital","bankrupt","time_to_ruin",
                    "ez_steps","ez_phases","ez_repeats",
                    "exploration_steps","exploration_rate",
                ]
                writer = csv.DictWriter(tf, fieldnames=fieldnames)
                writer.writeheader()
                for row in train_rows:
                    writer.writerow(row)

            # save evaluation rows if any
            if eval_rows:
                eval_file = results_dir / f"eval_metrics_{label}_{seed}.csv"
                with eval_file.open("w", newline="") as ef:
                    writer = csv.DictWriter(ef, fieldnames=[
                        "block_episode","variant","seed","episode","return",
                        "bankrupt","time_to_ruin","min_capital","final_capital",
                        "liabilities_end","terminal_paid",
                    ])
                    writer.writeheader()
                    for row in eval_rows:
                        writer.writerow(row)

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
