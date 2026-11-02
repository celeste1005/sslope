#!/usr/bin/env python3
"""Evaluate whether TA-01 test-model rankings survive coarser output rounding."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
source = json.loads((ROOT / "data/generated/ta01-fem-500-manifest.json").read_text(encoding="utf-8"))
date_by_scenario = {item["scenarioId"]: item["sourceDate"] for item in source["scenarios"]}
steps = (1e-5, 2e-5, 5e-5, 1e-4)
rng = np.random.default_rng(20261102)


def rows(path: Path, field: str):
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario_id"], int(row["origin_hour"]), int(row["target_hour"]))
            result[key] = (float(row["actual_displacement_mm"]), float(row[field]))
    return result


def pair(test_path_prefix: str, horizon: int, hybrid_type: str):
    base = f"data/generated/ta01{('-' + test_path_prefix) if test_path_prefix else ''}"
    lstm_path = ROOT / f"{base}-lstm-{horizon}h-test-predictions.csv"
    hybrid_suffix = f"-physics-guided-{horizon}h"
    hybrid_path = ROOT / f"{base}{hybrid_suffix}{'-oof' if hybrid_type == 'oof' else ''}-test-predictions.csv"
    lstm = rows(lstm_path, "lstm_prediction_mm")
    hybrid = rows(hybrid_path, "physics_guided_prediction_mm")
    if lstm.keys() != hybrid.keys():
        raise ValueError(f"Predicciones no alineadas: {hybrid_path}")
    raw = []
    for key, (actual, lstm_pred) in lstm.items():
        hybrid_actual, hybrid_pred = hybrid[key]
        if actual != hybrid_actual:
            raise ValueError(f"Objetivos distintos: {key}")
        raw.append((key, actual, lstm_pred, hybrid_pred))
    return raw


result = {
    "method": "Round observed targets and both model predictions to the same grid before computing window MAE; paired bootstrap by source date, 20,000 draws.",
    "roundingStepsMm": list(steps),
    "interpretation": "Positive LSTM-minus-hybrid differences favor the hybrid. Rounding is a sensitivity analysis of stored outputs, not a rerun of the FEM solver or a field-instrument precision study.",
    "bySplit": {},
}

for split in ("primary", "backtest-2024", "chronological"):
    prefix = "" if split == "primary" else split
    result["bySplit"][split] = {}
    for horizon in (1, 6):
        result["bySplit"][split][str(horizon)] = {}
        for hybrid_type in ("in-sample", "oof"):
            raw = pair(prefix, horizon, hybrid_type)
            entry = {"nWindows": len(raw), "nDates": len({date_by_scenario[item[0][0]] for item in raw}), "byStepMm": {}}
            for step in steps:
                grouped = defaultdict(lambda: {"lstm": [], "hybrid": []})
                all_lstm, all_hybrid = [], []
                for key, actual, lp, hp in raw:
                    qa = np.round(actual / step) * step
                    ql = np.round(lp / step) * step
                    qh = np.round(hp / step) * step
                    all_lstm.append(abs(ql - qa))
                    all_hybrid.append(abs(qh - qa))
                    date = date_by_scenario[key[0]]
                    grouped[date]["lstm"].append(abs(ql - qa))
                    grouped[date]["hybrid"].append(abs(qh - qa))
                dates = sorted(grouped)
                diff = np.asarray([
                    np.mean(grouped[d]["lstm"]) - np.mean(grouped[d]["hybrid"])
                    for d in dates
                ])
                boot = np.asarray([
                    diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(20000)
                ])
                entry["byStepMm"][f"{step:.0e}"] = {
                    "windowMaeMm": {"lstm": float(np.mean(all_lstm)), "hybrid": float(np.mean(all_hybrid))},
                    "pairedDateDifferenceMeanMm": float(diff.mean()),
                    "pairedDateBootstrap95Mm": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                    "hybridLowerWindowMae": bool(np.mean(all_hybrid) < np.mean(all_lstm)),
                }
            result["bySplit"][split][str(horizon)][hybrid_type] = entry

output = ROOT / "data/validation/ta01-rounding-sensitivity.json"
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
