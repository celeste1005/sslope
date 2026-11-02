#!/usr/bin/env python3
"""Hierarchical seed/date bootstrap for repeated primary-split OOF hybrids."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
manifest = json.loads((ROOT / "data/generated/ta01-fem-500-manifest.json").read_text(encoding="utf-8"))
date_by_scenario = {item["scenarioId"]: item["sourceDate"] for item in manifest["scenarios"]}
seeds = (20261029, 20261031, 20261032)
corrector_seeds = {20261029: 20261030, 20261031: 20261031, 20261032: 20261032}
rng = np.random.default_rng(20261101)


def read_rows(path: Path, prediction_field: str):
    output = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario_id"], int(row["origin_hour"]), int(row["target_hour"]))
            if key in output:
                raise ValueError(f"Clave duplicada {key}: {path}")
            actual = float(row["actual_displacement_mm"])
            prediction = float(row[prediction_field])
            output[key] = (actual, prediction)
    return output


summary = {}
for horizon in (1, 6):
    lstm = read_rows(
        ROOT / f"data/generated/ta01-lstm-{horizon}h-test-predictions.csv",
        "lstm_prediction_mm",
    )
    errors_by_seed_date = {}
    for seed in seeds:
        if seed == 20261029:
            path = ROOT / f"data/generated/ta01-physics-guided-{horizon}h-oof-test-predictions.csv"
        else:
            path = ROOT / f"data/generated/ta01-primary-physics-guided-{horizon}h-oof-s{seed}-test-predictions.csv"
        hybrid = read_rows(path, "physics_guided_prediction_mm")
        if hybrid.keys() != lstm.keys():
            raise ValueError(f"Ventanas distintas para h={horizon}, semilla={seed}")
        by_date = defaultdict(lambda: {"lstm": [], "hybrid": []})
        for key, (actual, base_prediction) in lstm.items():
            hybrid_actual, hybrid_prediction = hybrid[key]
            if actual != hybrid_actual:
                raise ValueError(f"Observado distinto en {key}")
            date = date_by_scenario[key[0]]
            by_date[date]["lstm"].append(abs(base_prediction - actual))
            by_date[date]["hybrid"].append(abs(hybrid_prediction - actual))
        errors_by_seed_date[seed] = by_date

    dates = sorted(next(iter(errors_by_seed_date.values())))
    differences = np.asarray([
        [
            np.mean(errors_by_seed_date[seed][date]["lstm"])
            - np.mean(errors_by_seed_date[seed][date]["hybrid"])
            for date in dates
        ]
        for seed in seeds
    ])
    paired_mae = differences.mean(axis=1)
    boot = []
    for _ in range(20000):
        sampled_seeds = rng.integers(0, len(seeds), len(seeds))
        sampled_dates = rng.integers(0, len(dates), len(dates))
        seed_means = []
        for index in sampled_seeds:
            seed_means.append(differences[index, sampled_dates].mean())
        boot.append(np.mean(seed_means))
    boot = np.asarray(boot)
    lstm_seed_maes, hybrid_seed_maes = [], []
    for seed in seeds:
        lstm_seed_maes.append(float(np.mean([
            np.mean(errors_by_seed_date[seed][date]["lstm"]) for date in dates
        ])))
        hybrid_seed_maes.append(float(np.mean([
            np.mean(errors_by_seed_date[seed][date]["hybrid"]) for date in dates
        ])))
    summary[str(horizon)] = {
        "seeds": list(seeds),
        "nRainfallDates": len(dates),
        "nWindows": len(lstm),
        "maeMmBySeed": {
            "lstm": lstm_seed_maes,
            "oofHybrid": hybrid_seed_maes,
        },
        "meanMaeMmAcrossSeeds": {
            "lstm": float(np.mean(lstm_seed_maes)),
            "oofHybrid": float(np.mean(hybrid_seed_maes)),
        },
        "sampleSdMaeMmAcrossSeeds": {
            "lstm": float(np.std(lstm_seed_maes, ddof=1)),
            "oofHybrid": float(np.std(hybrid_seed_maes, ddof=1)),
        },
        "pairedLstmMinusHybridMeanMm": float(paired_mae.mean()),
        "twoStageBootstrap95Mm": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "seedsWhereHybridHasLowerMae": int(np.sum(np.asarray(hybrid_seed_maes) < np.asarray(lstm_seed_maes))),
        "bootstrap": "20,000 resamples; seeds and dates resampled in two stages, seed 20261101.",
    }

result = {
    "method": "Repeated five-fold OOF LSTM predictions grouped by rainfall date; paired two-stage bootstrap over seed and shared resampled test dates.",
    "scope": "Exploratory primary-split sensitivity using three OOF seeds and one fixed final LSTM per horizon.",
    "replicates": [
        {"oofFoldSeed": seed, "correctorSeed": corrector_seeds[seed]}
        for seed in seeds
    ],
    "byHorizon": summary,
}
output = ROOT / "data/validation/ta01-oof-seed-sensitivity.json"
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
