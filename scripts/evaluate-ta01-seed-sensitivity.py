#!/usr/bin/env python3
"""Summarize TA-01 five-seed model sensitivity with paired date bootstrap."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SEEDS = [20260916, 20260918, 20260919, 20260920, 20260921]
manifest = json.loads((ROOT / "data/generated/ta01-fem-500-manifest.json").read_text(encoding="utf-8"))
date_by_scenario = {item["scenarioId"]: item["sourceDate"] for item in manifest["scenarios"]}
rng = np.random.default_rng(20261029)
report = {
    "method": "Two-stage paired bootstrap: resample five training-seed pairs and 75 test rainfall dates; 20,000 draws; seed 20261029.",
    "scope": "Exploratory initialization sensitivity on the primary date-grouped split. The four supplemental seed runs use the same train/validation/test scenarios and hyperparameters as the primary run.",
    "byHorizon": {},
}
csv_rows = []


def paths(horizon: int, seed: int):
    tag = f"{horizon}h-s{seed}"
    return (
        ROOT / f"data/models/ta01-seed-sensitivity-lstm-{tag}.json",
        ROOT / f"data/models/ta01-seed-sensitivity-physics-{tag}.json",
        ROOT / f"data/generated/ta01-seed-sensitivity-lstm-{tag}-predictions.csv",
        ROOT / f"data/generated/ta01-seed-sensitivity-physics-{tag}-predictions.csv",
    )


def date_mae(path: Path, prediction_field: str):
    grouped = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            date = date_by_scenario[row["scenario_id"]]
            grouped[date].append(abs(float(row[prediction_field]) - float(row["actual_displacement_mm"])))
    return {date: float(np.mean(errors)) for date, errors in grouped.items()}


for horizon in (1, 6):
    pairs = []
    date_differences = []
    for seed in SEEDS:
        lstm_path, hybrid_path, lstm_predictions_path, hybrid_predictions_path = paths(horizon, seed)
        lstm_artifact = json.loads(lstm_path.read_text(encoding="utf-8"))
        hybrid_artifact = json.loads(hybrid_path.read_text(encoding="utf-8"))
        if lstm_artifact["training"]["seed"] != seed:
            raise ValueError(f"Semilla inesperada en {lstm_path}")
        expected_corrector_seed = 20260917 if seed == 20260916 else seed + 1
        if hybrid_artifact["training"]["seed"] != expected_corrector_seed:
            raise ValueError(f"Semilla inesperada en {hybrid_path}")
        lstm_dates = date_mae(lstm_predictions_path, "lstm_prediction_mm")
        hybrid_dates = date_mae(hybrid_predictions_path, "physics_guided_prediction_mm")
        if lstm_dates.keys() != hybrid_dates.keys() or len(lstm_dates) != 75:
            raise ValueError(f"Ventanas/fechas no comparables para {horizon} h, seed {seed}")
        delta_by_date = {date: lstm_dates[date] - hybrid_dates[date] for date in lstm_dates}
        delta_by_date = dict(sorted(delta_by_date.items()))
        pairs.append({
            "seed": seed,
            "lstmMaeMm": float(lstm_artifact["metrics"]["test"]["lstm"]["maeMm"]),
            "hybridMaeMm": float(hybrid_artifact["metrics"]["test"]["pinn"]["maeMm"]),
            "lstmMinusHybridMaeMm": float(lstm_artifact["metrics"]["test"]["lstm"]["maeMm"] - hybrid_artifact["metrics"]["test"]["pinn"]["maeMm"]),
        })
        date_differences.append(delta_by_date)

    dates = sorted(date_differences[0])
    values = np.asarray([[run[date] for date in dates] for run in date_differences], dtype=float)
    bootstrap = np.empty(20000, dtype=float)
    for draw in range(bootstrap.size):
        seed_indices = rng.integers(0, len(SEEDS), len(SEEDS))
        date_indices = rng.integers(0, len(dates), len(dates))
        bootstrap[draw] = values[np.ix_(seed_indices, date_indices)].mean()
    mean_delta = float(values.mean())
    report["byHorizon"][str(horizon)] = {
        "seeds": pairs,
        "nRainfallDates": len(dates),
        "hybridWinCount": sum(pair["hybridMaeMm"] < pair["lstmMaeMm"] for pair in pairs),
        "lstmMaeMeanSdMm": {
            "mean": mean([pair["lstmMaeMm"] for pair in pairs]),
            "sampleSd": stdev([pair["lstmMaeMm"] for pair in pairs]),
        },
        "hybridMaeMeanSdMm": {
            "mean": mean([pair["hybridMaeMm"] for pair in pairs]),
            "sampleSd": stdev([pair["hybridMaeMm"] for pair in pairs]),
        },
        "lstmMinusHybridMeanMaeMm": mean_delta,
        "lstmMinusHybridSeedDateBootstrap95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "lstmMinusHybridSeedMeanSdMm": stdev([pair["lstmMinusHybridMaeMm"] for pair in pairs]),
    }
    for pair in pairs:
        csv_rows.append({"horizonHours": horizon, **pair, "hybridWins": pair["hybridMaeMm"] < pair["lstmMaeMm"]})

(ROOT / "data/validation/ta01-seed-sensitivity-summary.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
with (ROOT / "data/validation/ta01-seed-sensitivity.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(csv_rows)
print(json.dumps(report["byHorizon"], indent=2))
