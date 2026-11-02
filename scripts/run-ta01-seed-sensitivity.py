#!/usr/bin/env python3
"""Run paired LSTM and residual-corrector training for a fixed seed set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEEDS = [20260916, 20260918, 20260919, 20260920, 20260921]


def run(command: list[str]):
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--horizons", default="1,6")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    horizons = [int(value) for value in args.horizons.split(",") if value]
    if not seeds or any(h not in (1, 6) for h in horizons):
        raise ValueError("Use one or more seeds and horizons 1 or 6")

    python = sys.executable
    for horizon in horizons:
        for seed in seeds:
            tag = f"{horizon}h-s{seed}"
            lstm_model = f"data/models/ta01-seed-sensitivity-lstm-{tag}.json"
            lstm_predictions = f"data/generated/ta01-seed-sensitivity-lstm-{tag}-predictions.csv"
            corrector_model = f"data/models/ta01-seed-sensitivity-physics-{tag}.json"
            corrector_predictions = f"data/generated/ta01-seed-sensitivity-physics-{tag}-predictions.csv"
            run([
                python, "scripts/train-lstm.py", f"--horizon={horizon}", f"--seed={seed}",
                f"--output={lstm_model}", f"--predictions={lstm_predictions}",
            ])
            run([
                python, "scripts/train-physics-guided.py", f"--horizon={horizon}", f"--seed={seed + 1}",
                f"--lstm-model={lstm_model}", f"--output={corrector_model}", f"--predictions={corrector_predictions}",
            ])


if __name__ == "__main__":
    main()
