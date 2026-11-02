#!/usr/bin/env python3
"""Compare complete primary OOF pipelines after retraining at FEM tolerance 1e-6."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
manifest = json.loads((ROOT / "data/generated/ta01-fem-500-manifest.json").read_text(encoding="utf-8"))
date_by_scenario = {item["scenarioId"]: item["sourceDate"] for item in manifest["scenarios"]}
rng = np.random.default_rng(20261034)


def read_predictions(path: Path, lstm_field: str, hybrid_field: str):
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario_id"], int(row["origin_hour"]), int(row["target_hour"]))
            if key in result:
                raise ValueError(f"Ventana duplicada: {key} ({path})")
            result[key] = {
                "actual": float(row["actual_displacement_mm"]),
                "lstm": float(row[lstm_field]),
                "hybrid": float(row[hybrid_field]),
            }
    return result


def read_seed_predictions(horizon: int, seed: int, variant: bool):
    if variant:
        suffix = "" if seed == 20261029 else f"-s{seed}"
        path = ROOT / f"data/validation/ta01-tol1e-6-hybrid-{horizon}h{suffix}-predictions.csv"
    else:
        path = ROOT / (
            f"data/generated/ta01-physics-guided-{horizon}h-oof-test-predictions.csv"
            if seed == 20261029
            else f"data/generated/ta01-primary-physics-guided-{horizon}h-oof-s{seed}-test-predictions.csv"
        )
    return read_predictions(path, "lstm_prediction_mm", "physics_guided_prediction_mm")


def paired_summary(seed_rows: dict[int, dict], horizon: int):
    seeds = sorted(seed_rows)
    key_sets = [set(seed_rows[seed]) for seed in seeds]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError(f"Las semillas no comparten las mismas ventanas para {horizon} h")
    keys = sorted(key_sets[0])
    dates = sorted({date_by_scenario[key[0]] for key in keys})
    date_index = {date: index for index, date in enumerate(dates)}
    seed_date_deltas = np.zeros((len(seeds), len(dates)), dtype=float)
    per_seed = {}
    for seed_index, seed in enumerate(seeds):
        by_date = defaultdict(lambda: {"lstm": [], "hybrid": []})
        rows = seed_rows[seed]
        for key, row in rows.items():
            date = date_by_scenario[key[0]]
            by_date[date]["lstm"].append(abs(row["lstm"] - row["actual"]))
            by_date[date]["hybrid"].append(abs(row["hybrid"] - row["actual"]))
        deltas = []
        for date in dates:
            if date not in by_date:
                raise ValueError(f"Fecha compartida ausente: {date}")
            delta = np.mean(by_date[date]["lstm"]) - np.mean(by_date[date]["hybrid"])
            seed_date_deltas[seed_index, date_index[date]] = delta
            deltas.append(delta)
        per_seed[str(seed)] = {
            "lstmMaeMm": float(np.mean([abs(r["lstm"] - r["actual"]) for r in rows.values()])),
            "hybridMaeMm": float(np.mean([abs(r["hybrid"] - r["actual"]) for r in rows.values()])),
            "lstmMinusHybridMeanDateMaeMm": float(np.mean(deltas)),
        }
    draws = np.empty(20000, dtype=float)
    for i in range(draws.size):
        sampled_seeds = rng.integers(0, len(seeds), len(seeds))
        sampled_dates = rng.integers(0, len(dates), len(dates))
        draws[i] = seed_date_deltas[np.ix_(sampled_seeds, sampled_dates)].mean()
    return {
        "nSeeds": len(seeds),
        "seeds": seeds,
        "nWindowsPerSeed": len(keys),
        "nRainfallDates": len(dates),
        "seedResults": per_seed,
        "meanLstmMaeAcrossSeedsMm": float(np.mean([v["lstmMaeMm"] for v in per_seed.values()])),
        "meanHybridMaeAcrossSeedsMm": float(np.mean([v["hybridMaeMm"] for v in per_seed.values()])),
        "meanDateWeightedLstmMinusHybridMaeMm": float(seed_date_deltas.mean()),
        "pairedTwoStageBootstrap95Mm": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "seedsHybridHasLowerMae": int(sum(v["lstmMaeMm"] > v["hybridMaeMm"] for v in per_seed.values())),
    }


summary = {}
for horizon in (1, 6):
    seeds = (20261029, 20261031, 20261032)
    base = {seed: read_seed_predictions(horizon, seed, False) for seed in seeds}
    retrained = {seed: read_seed_predictions(horizon, seed, True) for seed in seeds}
    for seed in seeds:
        if base[seed].keys() != retrained[seed].keys():
            raise ValueError(f"La partición base y la reentrenada no comparten ventanas para {horizon} h, semilla {seed}")
    summary[str(horizon)] = {
        "baseTolerance1e-8": paired_summary(base, horizon),
        "retrainedTolerance1e-6": paired_summary(retrained, horizon),
    }

out = {
    "method": "Primary date-grouped split; same scenario/date split, same final LSTM seed (20260916), and three matched OOF fold/corrector seed pairs (20261029/20261030, 20261031/20261031, 20261032/20261032). For each pair, train-fold LSTMs and OOF hybrid corrector were retrained at relative FEM solver tolerance 1e-6. Paired two-stage bootstrap resamples the three matched seeds and 75 shared rainfall dates (20,000 draws; seed 20261034).",
    "matchedTrainingSeeds": {"finalLstm": 20260916, "oofFoldCorrectorPairs": [[20261029, 20261030], [20261031, 20261031], [20261032, 20261032]], "bootstrap": 20261034},
    "deltaSign": "Positive LSTM-minus-hybrid values mean the hybrid has lower MAE.",
    "scope": "One final LSTM initialization per horizon, one date-grouped split, and three OOF fold/corrector seed pairs; not external field validation.",
    "byHorizon": summary,
}
path = ROOT / "data/validation/ta01-solver-tolerance-retrained-comparison.json"
path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2))
