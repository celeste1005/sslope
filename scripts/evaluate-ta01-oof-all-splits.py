#!/usr/bin/env python3
"""Paired-by-date OOF corrector comparisons across TA-01 temporal splits."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
source_manifest = json.loads(
    (ROOT / "data/generated/ta01-fem-500-manifest.json").read_text(encoding="utf-8")
)
date_by_scenario = {
    item["scenarioId"]: item["sourceDate"] for item in source_manifest["scenarios"]
}
rng = np.random.default_rng(20261031)


def read_predictions(path: Path, field: str) -> dict[tuple[str, int, int], tuple[float, float]]:
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario_id"], int(row["origin_hour"]), int(row["target_hour"]))
            if key in rows:
                raise ValueError(f"Predicción duplicada: {key} ({path})")
            rows[key] = (float(row["actual_displacement_mm"]), float(row[field]))
    return rows


def compare(prefix: str, horizon: int) -> dict:
    base = "data/generated/ta01" + (f"-{prefix}" if prefix else "")
    lstm = read_predictions(
        ROOT / f"{base}-lstm-{horizon}h-test-predictions.csv", "lstm_prediction_mm"
    )
    oof = read_predictions(
        ROOT / f"{base}-physics-guided-{horizon}h-oof-test-predictions.csv",
        "physics_guided_prediction_mm",
    )
    in_sample = read_predictions(
        ROOT / f"{base}-physics-guided-{horizon}h-test-predictions.csv",
        "physics_guided_prediction_mm",
    )
    if lstm.keys() != oof.keys() or lstm.keys() != in_sample.keys():
        raise ValueError(f"Las ventanas no coinciden: {prefix}, horizonte {horizon} h")

    by_date: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"lstm": [], "oofHybrid": [], "inSampleHybrid": []}
    )
    for key, (actual, lstm_prediction) in lstm.items():
        date = date_by_scenario[key[0]]
        oof_actual, oof_prediction = oof[key]
        in_actual, in_prediction = in_sample[key]
        if actual != oof_actual or actual != in_actual:
            raise ValueError(f"Los valores observados no coinciden para {key}")
        by_date[date]["lstm"].append(abs(lstm_prediction - actual))
        by_date[date]["oofHybrid"].append(abs(oof_prediction - actual))
        by_date[date]["inSampleHybrid"].append(abs(in_prediction - actual))

    dates = sorted(by_date)
    maes = {
        name: np.asarray([np.mean(by_date[d][name]) for d in dates], dtype=float)
        for name in ("lstm", "oofHybrid", "inSampleHybrid")
    }
    window_maes = {
        name: float(sum(sum(by_date[d][name]) for d in dates) / len(lstm))
        for name in ("lstm", "oofHybrid", "inSampleHybrid")
    }
    record = {
        "nWindows": len(lstm),
        "nRainfallDates": len(dates),
        "maeMm": window_maes,
        "meanDateMaeMm": {name: float(values.mean()) for name, values in maes.items()},
        "pairedDifferencesMm": {},
    }
    for first, second, label in (
        ("lstm", "oofHybrid", "LSTM_minus_OOF_hybrid"),
        ("inSampleHybrid", "oofHybrid", "in_sample_hybrid_minus_OOF_hybrid"),
    ):
        diffs = maes[first] - maes[second]
        bootstrap = np.asarray([
            diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(20000)
        ])
        record["pairedDifferencesMm"][label] = {
            "mean": float(diffs.mean()),
            "bootstrap95": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "datesWhereFirstModelHasHigherMae": int(np.sum(diffs > 0)),
            "datesWhereSecondModelHasHigherOrEqualMae": int(np.sum(diffs <= 0)),
        }
    return record


result = {
    "method": "Paired bootstrap by test rainfall date, 20,000 draws per comparison, seed 20261031.",
    "deltaSign": "Positive means the first model has higher MAE than the second model.",
    "scope": "Exploratory single-seed OOF corrector evaluation across primary and chronological holdouts.",
    "splits": {
        split: {str(h): compare(split, h) for h in (1, 6)}
        for split in ("", "backtest-2024", "chronological")
    },
}
output = ROOT / "data/validation/ta01-oof-all-splits-comparison.json"
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
