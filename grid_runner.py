import itertools
import subprocess
import sys
from pathlib import Path

# --------------- Settings ---------------
RUN_EXPERIMENT = Path("run_experiment.py")

GRID = {
    "episodes":   [1000],
    "horizon":    [2000],
    "k_repeat":   [2, 3, 4, 8, 12, 16],
    "eps":        [0.01, 0.05, 0.10, 0.20, 0.30],
    "alpha":      [1e-4, 3e-4, 1e-3, 3e-3],
    "gamma":      [0.97, 0.99, 0.995, 0.999],
    "scale":      [0, 1],
    "seed":       [10],
    "delay_min":  [1, 3, 4],
    "delay_max":  [6, 10, 15, 20, 25],
}

# --------------- Helpers ---------------
def product_dict(d):
    keys = list(d.keys())
    for vals in itertools.product(*(d[k] for k in keys)):
        yield dict(zip(keys, vals))

def make_cmd(p):
    return [
        sys.executable, str(RUN_EXPERIMENT),
        "--episodes", str(p["episodes"]),
        "--horizon", str(p["horizon"]),
        "--k_repeat", str(p["k_repeat"]),
        "--eps", str(p["eps"]),
        "--alpha", str(p["alpha"]),
        "--gamma", str(p["gamma"]),
        "--seed", str(p["seed"]),
        "--scale", str(p["scale"]),
        "--delay_min", str(p["delay_min"]),
        "--delay_max", str(p["delay_max"]),
    ]

# --------------- Main ---------------
def main():
    if not RUN_EXPERIMENT.exists():
        print(f"Could not find {RUN_EXPERIMENT}. Run this from the repo root.", file=sys.stderr)
        sys.exit(1)

    for params in product_dict(GRID):
        cmd = make_cmd(params)
        # Falls du sehen willst, was ausgeführt wird, kurz entkommentieren:
        print(" ".join(cmd))
        subprocess.run(cmd, check=False)  # stoppt sofort, wenn ein Run fehlschlägt

if __name__ == "__main__":
    main()
