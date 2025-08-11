# sweep.py
import sys
import subprocess
from itertools import product
from pathlib import Path

PY = sys.executable
SCRIPT = Path(__file__).parent / "run_experiment.py"

grid = {
    "--episodes": [500],
    "--horizon": [1000],
}

keys = list(grid)

for combo in product(*(grid[k] for k in keys)):
    args = [PY, str(SCRIPT)]
    for k, v in zip(keys, combo):
        args += [k, str(v)]
    # Ausgaben werden direkt durchgereicht (kein Logging, kein Capture)
    subprocess.run(args, check=True)
