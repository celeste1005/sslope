#!/usr/bin/env python3
"""Compare TA-01 FEM datasets generated with different PCG tolerances."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data/generated/ta01-fem-500.csv"
VARIANTS = {
    "1e-6": ROOT / "data/validation/ta01-fem-500-tol1e-6.csv",
    "1e-10": ROOT / "data/validation/ta01-fem-500-tol1e-10.csv",
}
FIELDS = (
    "rainfall_induced_max_displacement_mm",
    "gravity_total_max_displacement_mm",
    "crest_displacement_mm",
    "toe_displacement_mm",
    "maximum_pore_pressure_kpa",
    "mean_effective_stress_kpa",
    "mohr_coulomb_safety_index",
)


def read(path: Path):
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["scenario_id"], int(row["simulation_hour"]))
            if key in rows:
                raise ValueError(f"Fila duplicada {key}: {path}")
            rows[key] = row
    return rows


def read_target(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            (row["scenario_id"], int(row["simulation_hour"])): float(row["rainfall_induced_max_displacement_mm"])
            for row in csv.DictReader(handle)
        }


baseline = read(BASE)
baseline_target = {
    key: float(row["rainfall_induced_max_displacement_mm"])
    for key, row in baseline.items()
}
summary = {}
for label, path in VARIANTS.items():
    variant = read(path)
    if baseline.keys() != variant.keys():
        raise ValueError(f"Las llaves escenario/hora no coinciden: {label}")
    fields = {}
    for name in FIELDS:
        before = np.asarray([float(baseline[key][name]) for key in baseline])
        after = np.asarray([float(variant[key][name]) for key in baseline])
        difference = after - before
        fields[name] = {
            "maxAbsoluteDifference": float(np.max(np.abs(difference))),
            "meanAbsoluteDifference": float(np.mean(np.abs(difference))),
            "changedRows": int(np.sum(difference != 0)),
            "maxRelativeDifference": float(np.max(np.abs(difference) / np.maximum(np.abs(before), 1e-30))),
        }
    residual_before = np.asarray([float(row["solver_residual"]) for row in baseline.values()])
    residual_after = np.asarray([float(row["solver_residual"]) for row in variant.values()])
    summary[label] = {
        "nRows": len(baseline),
        "baseTolerance": 1e-8,
        "variantTolerance": float(label),
        "fields": fields,
        "solverResidual": {
            "baseMaximum": float(np.max(residual_before)),
            "variantMaximum": float(np.max(residual_after)),
            "baseMean": float(np.mean(residual_before)),
            "variantMean": float(np.mean(residual_after)),
        },
    }
    variant_target = {
        key: float(row["rainfall_induced_max_displacement_mm"])
        for key, row in variant.items()
    }
    summary[label]["testPredictionSensitivity"] = {}
    for split in ("primary", "backtest-2024", "chronological"):
        prefix = "ta01" if split == "primary" else f"ta01-{split}"
        summary[label]["testPredictionSensitivity"][split] = {}
        for horizon in (1, 6):
            paths = {
                "lstm": (ROOT / f"data/generated/{prefix}-lstm-{horizon}h-test-predictions.csv", "lstm_prediction_mm"),
                "inSampleHybrid": (ROOT / f"data/generated/{prefix}-physics-guided-{horizon}h-test-predictions.csv", "physics_guided_prediction_mm"),
                "oofHybrid": (ROOT / f"data/generated/{prefix}-physics-guided-{horizon}h-oof-test-predictions.csv", "physics_guided_prediction_mm"),
            }
            comparison = {}
            for model_name, (path, prediction_field) in paths.items():
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    errors_base, errors_variant = [], []
                    for row in csv.DictReader(handle):
                        target_key = (row["scenario_id"], int(row["target_hour"]))
                        prediction = float(row[prediction_field])
                        baseline_actual = float(row["actual_displacement_mm"])
                        variant_actual = variant_target[target_key]
                        if baseline_actual != baseline_target[target_key]:
                            raise ValueError(f"MAE base no coincide: {target_key}")
                        errors_base.append(abs(prediction - baseline_actual))
                        errors_variant.append(abs(prediction - variant_actual))
                comparison[model_name] = {
                    "baseMaeMm": float(np.mean(errors_base)),
                    "variantTargetMaeMm": float(np.mean(errors_variant)),
                    "maeChangeMm": float(np.mean(errors_variant) - np.mean(errors_base)),
                }
            summary[label]["testPredictionSensitivity"][split][str(horizon)] = comparison

result = {
    "method": "Regenerate the identical 500 LHS scenarios (seed 20260915, mesh 30x20) while changing only the PCG relative tolerance; compare saved 24-hour FEM outputs by scenario and hour.",
    "base": "data/generated/ta01-fem-500.csv, default PCG relative tolerance 1e-8.",
    "variants": summary,
    "limitations": "This checks numerical sensitivity of regenerated FEM outputs, not retrained predictive-model performance, field measurement accuracy, or mesh/model-form uncertainty.",
}
output = ROOT / "data/validation/ta01-solver-tolerance-sensitivity.json"
output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
