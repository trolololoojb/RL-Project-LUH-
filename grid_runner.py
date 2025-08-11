#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Random-search orchestrator for run_experiment.py
# - Phase 1: sample N random configs from ranges close to the current grid values.
# - Phase 2: take Top-K (by EZ tail-mean) + a safety config and train longer.

import csv
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

RUN_EXPERIMENT = Path("run_experiment.py")
RESULTS_ROOT = Path("results")

# ----------- Controls -----------
PHASE1_TRIALS     = 80      # number of random samples in phase 1 (adjust for runtime)
PHASE1_EPISODES   = 20
PHASE1_HORIZON    = 600
PHASE2_EPISODES   = 100
PHASE2_HORIZON    = 900
TOP_K             = 3
BASE_SEED         = 42      # used here AND passed into run_experiment (which also uses seed+1 internally)
SCALE_EZ          = 1
RNG               = random.Random(BASE_SEED)

# Conservative "safety" config (small k, cautious eps, high gamma)
SAFETY_CONF = dict(
    delay_min=3, delay_max=10,
    gamma=0.999,
    alpha=3e-4,
    eps=0.04,
    k_repeat=3,                  # small & safe for per-customer steps
    bankruptcy_penalty=250_000,
    hidden_dims="256,256",
    batch_size=128,
    sync_every=800,
    buffer_capacity=200_000,
)

@dataclass
class TrialResult:
    params: dict
    results_dir: str
    ez_tail_mean: float
    fixed_tail_mean: float

def _latest_results_dir():
    if not RESULTS_ROOT.exists():
        return None
    dirs = [p for p in RESULTS_ROOT.glob("*") if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)

def _run_and_collect(params: dict, episodes: int, horizon: int) -> TrialResult:
    args = [
        "--episodes", str(episodes),
        "--horizon", str(horizon),
        "--k_repeat", str(params["k_repeat"]),
        "--eps", str(params["eps"]),
        "--alpha", str(params["alpha"]),
        "--gamma", str(params["gamma"]),
        "--seed", str(BASE_SEED),
        "--scale", str(SCALE_EZ),
        "--delay_min", str(params["delay_min"]),
        "--delay_max", str(params["delay_max"]),
        "--bankruptcy_penalty", str(params["bankruptcy_penalty"]),
        "--hidden_dims", str(params["hidden_dims"]),
        "--batch_size", str(params["batch_size"]),
        "--sync_every", str(params["sync_every"]),
        "--buffer_capacity", str(params["buffer_capacity"]),
    ]
    cmd = [sys.executable, str(RUN_EXPERIMENT)] + args
    print("\n>>", " ".join(map(str, cmd)))
    before = _latest_results_dir()
    subprocess.run(cmd, check=False)
    time.sleep(0.5)
    after = _latest_results_dir()
    if after is None or before == after:
        after = _latest_results_dir()
    if after is None:
        raise RuntimeError("Could not locate results directory after run_experiment.")
    summary_csv = after / "experiment_summary.csv"
    if not summary_csv.exists():
        raise RuntimeError(f"Missing experiment_summary.csv in {after}")
    returns_by_variant = {"EZ": [], "Fixed": []}
    with summary_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    ep_numbers = [int(r["episode"]) for r in rows if r["variant"] == "EZ"]
    n_episodes = max(ep_numbers) if ep_numbers else episodes
    tail_start = max(1, int(n_episodes * (2/3)))
    for r in rows:
        ep = int(r["episode"])
        if ep >= tail_start:
            v = r["variant"]
            returns_by_variant[v].append(float(r["return"]))
    ez_tail = mean(returns_by_variant["EZ"]) if returns_by_variant["EZ"] else float("-inf")
    fixed_tail = mean(returns_by_variant["Fixed"]) if returns_by_variant["Fixed"] else float("-inf")
    print(f"   Tail means → EZ: {ez_tail:,.1f} | Fixed: {fixed_tail:,.1f} | dir={after.name}")
    return TrialResult(params=params, results_dir=str(after), ez_tail_mean=ez_tail, fixed_tail_mean=fixed_tail)

# ----- Random samplers oriented around current values -----

def _choice_weighted(items, weights):
    x = RNG.random() * sum(weights)
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if x <= acc:
            return it
    return items[-1]

def _sample_delay_pair():
    # stick to existing triplet, but favor shorter delays slightly
    pairs = [(3,10), (5,15), (8,25)]
    w = [0.5, 0.3, 0.2]
    return _choice_weighted(pairs, w)

def _sample_gamma():
    # bias toward 0.999
    candidates = [0.995, 0.997, 0.999]
    w = [0.2, 0.3, 0.5]
    return _choice_weighted(candidates, w)

def _sample_alpha():
    # log-uniform around [2.5e-4, 1.5e-3], centered near 3e-4 and 1e-3
    lo, hi = 2.5e-4, 1.5e-3
    u = RNG.random()
    val = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    # snap occasionally to the known good anchors
    if RNG.random() < 0.3:
        return _choice_weighted([3e-4, 1e-3], [0.6, 0.4])
    return val

def _sample_eps():
    # favor 0.04 and 0.08 but allow a small spread
    anchors = [0.03, 0.04, 0.06, 0.08, 0.10]
    w = [0.15, 0.35, 0.20, 0.25, 0.05]
    base = _choice_weighted(anchors, w)
    jitter = (RNG.random() - 0.5) * 0.01  # ±0.005
    return max(0.005, min(0.12, base + jitter))

def _sample_k():
    # keep small; bias to {2,3}, rare 6
    ks = [2, 3, 4, 6]
    w = [0.4, 0.35, 0.2, 0.05]
    return _choice_weighted(ks, w)

def _sample_sync_every():
    # around 800/1200 with small jitter
    anchors = [800, 1200]
    base = _choice_weighted(anchors, [0.6, 0.4])
    jitter = int(round((RNG.random()-0.5) * 200))  # ±100
    return max(300, base + jitter)

def _sample_penalty():
    return _choice_weighted([200_000, 250_000, 300_000], [0.4, 0.2, 0.4])

def _phase1_scenarios_random(n_trials: int):
    for _ in range(n_trials):
        dmin, dmax = _sample_delay_pair()
        yield dict(
            delay_min=dmin,
            delay_max=dmax,
            gamma=_sample_gamma(),
            alpha=_sample_alpha(),
            eps=_sample_eps(),
            k_repeat=_sample_k(),
            hidden_dims="256,256",
            batch_size=128,
            sync_every=_sample_sync_every(),
            buffer_capacity=200_000,
            bankruptcy_penalty=_sample_penalty(),
        )

def run():
    RNG.seed(BASE_SEED)
    if not RUN_EXPERIMENT.exists():
        print(f"ERR: Could not find {RUN_EXPERIMENT}. Run this from the repo root.", file=sys.stderr)
        sys.exit(1)

    # ---- Phase 1: random search ----
    tried = []
    for i, conf in enumerate(_phase1_scenarios_random(PHASE1_TRIALS), start=1):
        print(f"\n=== Phase 1 / {i}: {conf} ===")
        res = _run_and_collect(conf, PHASE1_EPISODES, PHASE1_HORIZON)
        tried.append(res)

    tried_sorted = sorted(tried, key=lambda r: r.ez_tail_mean, reverse=True)

    # ---- Phase 2: longer training on TOP_K + SAFETY ----
    finalists = [t.params for t in tried_sorted[:TOP_K]]
    if SAFETY_CONF not in finalists:
        finalists.append(SAFETY_CONF)

    print("\n=== Finalists for Phase 2 ===")
    for i, p in enumerate(finalists, 1):
        print(f"{i}. {p}")

    phase2_results = []
    for i, conf in enumerate(finalists, start=1):
        print(f"\n*** Phase 2 / {i}: {conf}")
        res = _run_and_collect(conf, PHASE2_EPISODES, PHASE2_HORIZON)
        phase2_results.append(res)

    # Save a summary json
    summary = {
        "phase1_top": [
            dict(params=r.params, ez_tail_mean=r.ez_tail_mean, fixed_tail_mean=r.fixed_tail_mean, dir=r.results_dir)
            for r in tried_sorted[:TOP_K]
        ],
        "phase2": [
            dict(params=r.params, ez_tail_mean=r.ez_tail_mean, fixed_tail_mean=r.fixed_tail_mean, dir=r.results_dir)
            for r in phase2_results
        ],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase1_trials": PHASE1_TRIALS,
    }
    out = Path("grid_summary_random.json")
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nAll done. Wrote summary → {out.resolve()}")

if __name__ == "__main__":
    run()
