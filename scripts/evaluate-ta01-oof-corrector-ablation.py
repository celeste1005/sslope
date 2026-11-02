#!/usr/bin/env python3
"""Compare monotonic and free residual correctors on matched primary OOF runs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "data/generated/ta01-fem-500-manifest.json").read_text(encoding="utf-8"))
DATE_BY_SCENARIO = {item["scenarioId"]: item["sourceDate"] for item in MANIFEST["scenarios"]}
SEED_PAIRS = [(20261029, 20261030), (20261031, 20261031), (20261032, 20261032)]


def read_prediction(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            key = (row["scenario_id"], int(row["origin_hour"]), int(row["target_hour"]))
            rows[key] = {
                "date": DATE_BY_SCENARIO[row["scenario_id"]],
                "actual": float(row["actual_displacement_mm"]),
                "raw": float(row["physics_guided_unconstrained_prediction_mm"]),
                "floored": float(row["physics_guided_prediction_mm"]),
            }
    return rows


def bootstrap(date_deltas: dict[str, np.ndarray], seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    seeds = sorted(date_deltas)
    date_names = sorted(set.intersection(*(set(date_deltas[s]) for s in seeds)))
    draws = np.empty(20000, dtype=float)
    for i in range(draws.size):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_dates = rng.choice(date_names, size=len(date_names), replace=True)
        per_seed = []
        for selected_seed in sampled_seeds:
            per_seed.append(float(np.mean([date_deltas[selected_seed][d] for d in sampled_dates])))
        draws[i] = float(np.mean(per_seed))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def summarize(horizon: int) -> dict:
    runs = []
    delta_by_date_raw = {}
    delta_by_date_floored = {}
    violations = {"rain": [], "safety": []}
    for fold_seed, corrector_seed in SEED_PAIRS:
        if fold_seed == 20261029:
            hard_path = ROOT / f"data/generated/ta01-physics-guided-{horizon}h-oof-test-predictions.csv"
        else:
            hard_path = ROOT / f"data/generated/ta01-primary-physics-guided-{horizon}h-oof-s{fold_seed}-test-predictions.csv"
        free_path = ROOT / f"data/validation/ta01-free-corrector-{horizon}h-s{fold_seed}-predictions.csv"
        hard = read_prediction(hard_path)
        free = read_prediction(free_path)
        if hard.keys() != free.keys():
            raise ValueError(f"Unmatched windows for horizon {horizon} and fold seed {fold_seed}")
        keys = sorted(hard)
        raw_delta = {}
        floored_delta = {}
        for key in keys:
            hrow, frow = hard[key], free[key]
            if hrow["actual"] != frow["actual"] or hrow["date"] != frow["date"]:
                raise ValueError(f"Mismatched target or date for {key}")
            date = hrow["date"]
            raw_delta.setdefault(date, []).append(abs(frow["raw"] - frow["actual"]) - abs(hrow["raw"] - hrow["actual"]))
            floored_delta.setdefault(date, []).append(abs(frow["floored"] - frow["actual"]) - abs(hrow["floored"] - hrow["actual"]))
        raw_delta = {d: float(np.mean(v)) for d, v in raw_delta.items()}
        floored_delta = {d: float(np.mean(v)) for d, v in floored_delta.items()}
        delta_by_date_raw[str(fold_seed)] = raw_delta
        delta_by_date_floored[str(fold_seed)] = floored_delta
        hard_metrics = json.loads((ROOT / (
            f"data/models/ta01-physics-guided-{horizon}h-oof.json" if fold_seed == 20261029 else
            f"data/models/ta01-primary-physics-guided-{horizon}h-oof-s{fold_seed}.json"
        )).read_text(encoding="utf-8"))["metrics"]["test"]
        free_metrics = json.loads((ROOT / f"data/models/ta01-free-corrector-{horizon}h-s{fold_seed}.json").read_text(encoding="utf-8"))["metrics"]["test"]
        runs.append({
            "foldSeed": fold_seed,
            "correctorSeed": corrector_seed,
            "nWindows": len(keys),
            "nDates": len(raw_delta),
            "monotonicMaeMm": hard_metrics["pinnUnconstrained"]["maeMm"],
            "freeMaeMm": free_metrics["pinnUnconstrained"]["maeMm"],
            "rawMaeDifferenceFreeMinusMonotonicMm": float(np.mean([raw_delta[d] for d in raw_delta])),
            "flooredMaeDifferenceFreeMinusMonotonicMm": float(np.mean([floored_delta[d] for d in floored_delta])),
            "freeRainSensitivityViolations": free_metrics["physicsChecks"]["rainSensitivityViolations"],
            "freeSafetySensitivityViolations": free_metrics["physicsChecks"]["safetySensitivityViolations"],
        })
        violations["rain"].append(free_metrics["physicsChecks"]["rainSensitivityViolations"])
        violations["safety"].append(free_metrics["physicsChecks"]["safetySensitivityViolations"])
    raw_diffs = [run["rawMaeDifferenceFreeMinusMonotonicMm"] for run in runs]
    floored_diffs = [run["flooredMaeDifferenceFreeMinusMonotonicMm"] for run in runs]
    return {
        "nSeeds": len(runs),
        "nWindowsPerSeed": runs[0]["nWindows"],
        "nRainfallDatesPerSeed": runs[0]["nDates"],
        "metricDefinition": "Window-weighted MAE on raw corrector outputs before the common nondecreasing-displacement floor; positive free-minus-monotonic values favor the monotonic corrector.",
        "runs": runs,
        "meanRawMaeDifferenceFreeMinusMonotonicMm": float(np.mean(raw_diffs)),
        "pairedTwoStageRawBootstrap95Mm": bootstrap(delta_by_date_raw, 20261060 + horizon),
        "meanFlooredMaeDifferenceFreeMinusMonotonicMm": float(np.mean(floored_diffs)),
        "pairedTwoStageFlooredBootstrap95Mm": bootstrap(delta_by_date_floored, 20261070 + horizon),
        "freeMonotonicityViolationCounts": violations,
    }


result = {
    "method": "Exploratory architecture ablation on the primary date-grouped split. The final LSTM, folds, windows, corrector hidden width (16), training targets, common nondecreasing-displacement floor and seeds are matched; only the residual corrector parameterization is changed. The paired two-stage bootstrap resamples the same date indices across all selected seeds within each of 20,000 draws.",
    "scope": "One final LSTM initialization per horizon, three matched OOF fold/corrector seed pairs, and synthetic data only. Free mode has no hard monotonic guarantee; ablation does not establish field validity.",
    "byHorizon": {str(h): summarize(h) for h in (1, 6)},
}
output = ROOT / "data/validation/ta01-oof-corrector-ablation.json"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
