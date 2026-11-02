#!/usr/bin/env python3
"""Paired date-bootstrap comparison for the TA-01 out-of-fold corrector pilot."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
manifest = json.loads((ROOT / "data/generated/ta01-fem-500-manifest.json").read_text(encoding="utf-8"))
date_by_scenario = {item["scenarioId"]: item["sourceDate"] for item in manifest["scenarios"]}
summary = {}
rng = np.random.default_rng(20261030)


def read_predictions(path: Path, model_field: str):
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario_id"], int(row["origin_hour"]), int(row["target_hour"]))
            if key in rows:
                raise ValueError(f"Ventana de predicción duplicada: {key} ({path})")
            rows[key] = (float(row["actual_displacement_mm"]), float(row[model_field]))
    return rows


for horizon in (1, 6):
    lstm_rows = read_predictions(
        ROOT / f"data/generated/ta01-lstm-{horizon}h-test-predictions.csv", "lstm_prediction_mm"
    )
    in_sample_rows = read_predictions(
        ROOT / f"data/generated/ta01-physics-guided-{horizon}h-test-predictions.csv",
        "physics_guided_prediction_mm",
    )
    oof_rows = read_predictions(
        ROOT / f"data/generated/ta01-physics-guided-{horizon}h-oof-test-predictions.csv",
        "physics_guided_prediction_mm",
    )
    if lstm_rows.keys() != in_sample_rows.keys() or lstm_rows.keys() != oof_rows.keys():
        raise ValueError(f"Las predicciones no comparten las mismas ventanas para horizonte {horizon} h")

    errors_by_date = defaultdict(lambda: {"lstm": [], "inSampleHybrid": [], "oofHybrid": []})
    for key, (actual, lstm_prediction) in lstm_rows.items():
        date = date_by_scenario[key[0]]
        in_actual, in_prediction = in_sample_rows[key]
        oof_actual, oof_prediction = oof_rows[key]
        if actual != in_actual or actual != oof_actual:
            raise ValueError(f"Los objetivos no coinciden para {key}")
        errors_by_date[date]["lstm"].append(abs(lstm_prediction - actual))
        errors_by_date[date]["inSampleHybrid"].append(abs(in_prediction - actual))
        errors_by_date[date]["oofHybrid"].append(abs(oof_prediction - actual))

    dates = sorted(errors_by_date)
    mae_by_date = {
        name: np.asarray([np.mean(errors_by_date[date][name]) for date in dates], dtype=float)
        for name in ("lstm", "inSampleHybrid", "oofHybrid")
    }
    comparisons = {
        "LSTM_minus_OOF_hybrid": mae_by_date["lstm"] - mae_by_date["oofHybrid"],
        "in_sample_hybrid_minus_OOF_hybrid": mae_by_date["inSampleHybrid"] - mae_by_date["oofHybrid"],
    }
    horizon_record = {
        "nWindows": len(lstm_rows),
        "nRainfallDates": len(dates),
        "maeMm": {name: float(values.mean()) for name, values in mae_by_date.items()},
        "pairedDifferencesMm": {},
    }
    for name, values in comparisons.items():
        draws = np.asarray([
            values[rng.integers(0, len(values), len(values))].mean()
            for _ in range(20000)
        ])
        horizon_record["pairedDifferencesMm"][name] = {
            "mean": float(values.mean()),
            "bootstrap95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "datesWhereFirstModelHasHigherMae": int(np.sum(values > 0)),
            "datesWhereSecondModelHasHigherOrEqualMae": int(np.sum(values <= 0)),
        }
    summary[str(horizon)] = horizon_record

output = ROOT / "data/validation/ta01-oof-comparison-bootstrap.json"
output.write_text(json.dumps({
    "method": "Paired bootstrap by test rainfall date, 20,000 draws per horizon, seed 20261030.",
    "deltaSign": "Positive differences mean the first model has higher MAE than the second model.",
    "scope": "Exploratory single-seed sensitivity analysis for the primary date-grouped split.",
    "byHorizon": summary,
}, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
