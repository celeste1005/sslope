#!/usr/bin/env python3
"""Paired seed/date sensitivity of OOF correctors for all TA-01 splits."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
source = json.loads((ROOT / "data/generated/ta01-fem-500-manifest.json").read_text(encoding="utf-8"))
date_by_scenario = {item["scenarioId"]: item["sourceDate"] for item in source["scenarios"]}
seed_sets = {"primary": (20261029, 20261031, 20261032), "backtest-2024": (20261029, 20261031), "chronological": (20261029, 20261031)}
corrector_seed = {20261029: 20261030, 20261031: 20261031, 20261032: 20261032}
rng = np.random.default_rng(20261103)


def read(path: Path, field: str):
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario_id"], int(row["origin_hour"]), int(row["target_hour"]))
            if key in rows:
                raise ValueError(f"Clave duplicada {key}: {path}")
            rows[key] = (float(row["actual_displacement_mm"]), float(row[field]))
    return rows


result = {
    "method": "Five-fold OOF predictions by source date; paired two-stage bootstrap resampling seeds and the same test-date indices across selected seeds, 20,000 draws.",
    "scope": "Three seeds on the primary split and two seeds on each chronological split; final LSTM fixed per split and horizon.",
    "seedRuns": {split: [{"oofFoldSeed": seed, "correctorSeed": corrector_seed[seed]} for seed in seeds] for split, seeds in seed_sets.items()},
    "splits": {},
}
for split, seeds in seed_sets.items():
    prefix = "" if split == "primary" else f"-{split}"
    result["splits"][split] = {}
    for horizon in (1, 6):
        base = f"data/generated/ta01{prefix}"
        lstm = read(ROOT / f"{base}-lstm-{horizon}h-test-predictions.csv", "lstm_prediction_mm")
        errors = {}
        for seed in seeds:
            suffix = "-oof" if seed == 20261029 else f"-oof-s{seed}"
            hybrid_prefix = "data/generated/ta01-primary" if split == "primary" and seed != 20261029 else base
            hybrid = read(ROOT / f"{hybrid_prefix}-physics-guided-{horizon}h{suffix}-test-predictions.csv", "physics_guided_prediction_mm")
            if lstm.keys() != hybrid.keys():
                raise ValueError(f"Ventanas distintas: split={split}, h={horizon}, seed={seed}")
            by_date = defaultdict(lambda: {"lstm": [], "hybrid": []})
            for key, (actual, base_prediction) in lstm.items():
                hybrid_actual, hybrid_prediction = hybrid[key]
                if actual != hybrid_actual:
                    raise ValueError(f"Objetivo distinto para {key}")
                date = date_by_scenario[key[0]]
                by_date[date]["lstm"].append(abs(base_prediction - actual))
                by_date[date]["hybrid"].append(abs(hybrid_prediction - actual))
            errors[seed] = by_date
        dates = sorted(next(iter(errors.values())))
        diff = np.asarray([[np.mean(errors[s][d]["lstm"]) - np.mean(errors[s][d]["hybrid"]) for d in dates] for s in seeds])
        boot = np.empty(20000)
        for draw in range(boot.size):
            seed_indices = rng.integers(0, len(seeds), len(seeds))
            date_indices = rng.integers(0, len(dates), len(dates))
            boot[draw] = np.mean([diff[s, date_indices].mean() for s in seed_indices])
        lstm_mae, hybrid_mae = [], []
        for seed in seeds:
            lstm_mae.append(float(sum(sum(errors[seed][d]["lstm"]) for d in dates) / len(lstm)))
            hybrid_mae.append(float(sum(sum(errors[seed][d]["hybrid"]) for d in dates) / len(lstm)))
        result["splits"][split][str(horizon)] = {
            "nWindows": len(lstm), "nRainfallDates": len(dates),
            "maeMmBySeed": {"lstm": lstm_mae, "oofHybrid": hybrid_mae},
            "meanMaeMmAcrossSeeds": {"lstm": float(np.mean(lstm_mae)), "oofHybrid": float(np.mean(hybrid_mae))},
            "sampleSdMaeMmAcrossSeeds": {"lstm": float(np.std(lstm_mae, ddof=1)), "oofHybrid": float(np.std(hybrid_mae, ddof=1))},
            "pairedLstmMinusHybridMeanByDateMm": float(diff.mean()),
            "sharedDateTwoStageBootstrap95Mm": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            "seedsWhereHybridHasLowerWindowMae": int(np.sum(np.asarray(hybrid_mae) < np.asarray(lstm_mae))),
        }

output = ROOT / "data/validation/ta01-oof-seed-sensitivity-all-splits.json"
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
