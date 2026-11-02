#!/usr/bin/env python3
"""Describe OOF errors for stored-zero vs positive displacement targets."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
manifest = json.loads((ROOT / "data/generated/ta01-fem-500-manifest.json").read_text(encoding="utf-8"))
date_by_scenario = {item["scenarioId"]: item["sourceDate"] for item in manifest["scenarios"]}
SPLITS = {
    "primary": "ta01-physics-guided-{h}h-oof-test-predictions.csv",
    "backtest2024": "ta01-backtest-2024-physics-guided-{h}h-oof-test-predictions.csv",
    "chronological2025": "ta01-chronological-physics-guided-{h}h-oof-test-predictions.csv",
}


def percentile(values, q):
    return float(np.quantile(np.asarray(values, dtype=float), q))


def read_rows(path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            actual = float(row["actual_displacement_mm"])
            current = float(row["persistence_prediction_mm"])
            lstm = float(row["lstm_prediction_mm"])
            hybrid = float(row["physics_guided_prediction_mm"])
            rows.append({
                "date": date_by_scenario[row["scenario_id"]],
                "actual": actual,
                "current": current,
                "lstm": lstm,
                "hybrid": hybrid,
            })
    return rows


def summarize(rows, seed):
    actual = np.asarray([r["actual"] for r in rows])
    current = np.asarray([r["current"] for r in rows])
    result = {
        "nWindows": len(rows),
        "nRainfallDates": len({r["date"] for r in rows}),
        "storedTarget": {
            "zeroCount": int(np.sum(actual == 0.0)),
            "zeroShare": float(np.mean(actual == 0.0)),
            "positiveCount": int(np.sum(actual > 0.0)),
            "positiveShare": float(np.mean(actual > 0.0)),
            "quantilesMm": {str(q): percentile(actual, q) for q in (0.5, 0.9, 0.95, 0.99, 1.0)},
        },
        "changeFromCurrent": {
            "zeroCount": int(np.sum(actual == current)),
            "positiveCount": int(np.sum(actual > current)),
            "negativeCount": int(np.sum(actual < current)),
        },
        "byStoredTarget": {},
    }
    rng = np.random.default_rng(seed)
    for regime, predicate in (("zero", lambda a: a == 0.0), ("positive", lambda a: a > 0.0)):
        selected = predicate(actual)
        group = [r for r, keep in zip(rows, selected) if keep]
        by_date = defaultdict(lambda: {"lstm": [], "hybrid": []})
        for r in group:
            by_date[r["date"]]["lstm"].append(abs(r["lstm"] - r["actual"]))
            by_date[r["date"]]["hybrid"].append(abs(r["hybrid"] - r["actual"]))
        dates = sorted(by_date)
        date_deltas = np.asarray([
            np.mean(by_date[date]["lstm"]) - np.mean(by_date[date]["hybrid"])
            for date in dates
        ], dtype=float)
        draws = np.asarray([
            date_deltas[rng.integers(0, len(dates), len(dates))].mean()
            for _ in range(20000)
        ])
        result["byStoredTarget"][regime] = {
            "nWindows": len(group),
            "nDatesWithRegime": len(dates),
            "lstmMaeMm": float(np.mean([abs(r["lstm"] - r["actual"]) for r in group])),
            "oofHybridMaeMm": float(np.mean([abs(r["hybrid"] - r["actual"]) for r in group])),
            "dateWeightedLstmMinusHybridMaeMm": float(date_deltas.mean()),
            "pairedDateBootstrap95Mm": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        }
    return result


output = {
    "method": "Exploratory OOF stratification by the stored future target being exactly zero or positive; MAE is window-weighted within each regime. Paired uncertainty resamples rainfall dates (20,000 draws per regime; deterministic seeds by horizon and split). Target values were stored after FEM output quantization.",
    "interpretation": "A stored zero is not evidence of no physical displacement below the FEM/output resolution. These descriptive strata do not define operational events or a pre-specified hypothesis.",
    "bySplit": {},
}
for split_name, template in SPLITS.items():
    output["bySplit"][split_name] = {}
    for h in (1, 6):
        path = ROOT / "data/generated" / template.format(h=h)
        rows = read_rows(path)
        seed = 20261040 + h + (0 if split_name == "primary" else 10 if split_name == "backtest2024" else 20)
        output["bySplit"][split_name][str(h)] = summarize(rows, seed)

out_path = ROOT / "data/validation/ta01-oof-target-regime-summary.json"
out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(json.dumps(output, indent=2))
