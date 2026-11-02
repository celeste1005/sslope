#!/usr/bin/env python3
"""Evaluate a history-only proportional-rainfall baseline against archived TA-01 forecasts.

The slope is estimated through the origin from each six-hour input history:
    beta = sum(cumulative_rainfall * displacement) / sum(cumulative_rainfall**2)
The target forecast is beta times target-hour cumulative rainfall, floored at the
observed origin displacement, matching the other models' nondecreasing output rule.
No test target is used to fit beta or choose a hyperparameter.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
LOOKBACK = 6
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20261040

SPLITS = {
    "primary": {
        "rows": "data/generated/ta01-fem-500-test.csv",
        "lstm": "data/generated/ta01-lstm-{h}h-test-predictions.csv",
        "in_sample_hybrid": "data/generated/ta01-physics-guided-{h}h-test-predictions.csv",
        "oof_hybrid": "data/generated/ta01-physics-guided-{h}h-oof-test-predictions.csv",
    },
    "backtest-2024": {
        "rows": "data/generated/ta01-fem-500-backtest-2024-test.csv",
        "lstm": "data/generated/ta01-backtest-2024-lstm-{h}h-test-predictions.csv",
        "in_sample_hybrid": "data/generated/ta01-backtest-2024-physics-guided-{h}h-test-predictions.csv",
        "oof_hybrid": "data/generated/ta01-backtest-2024-physics-guided-{h}h-oof-test-predictions.csv",
    },
    "chronological-2025": {
        "rows": "data/generated/ta01-fem-500-chronological-test.csv",
        "lstm": "data/generated/ta01-chronological-lstm-{h}h-test-predictions.csv",
        "in_sample_hybrid": "data/generated/ta01-chronological-physics-guided-{h}h-test-predictions.csv",
        "oof_hybrid": "data/generated/ta01-chronological-physics-guided-{h}h-oof-test-predictions.csv",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(predicted - actual)))


def paired_date_bootstrap(
    actual: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    dates: np.ndarray,
    seed: int,
) -> dict:
    """Bootstrap date clusters; positive delta means candidate has higher MAE."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, date in enumerate(dates):
        grouped[str(date)].append(index)
    date_keys = sorted(grouped)
    if not date_keys:
        raise ValueError("No date groups for bootstrap")
    candidate_loss_by_date = np.asarray([
        np.abs(candidate[grouped[key]] - actual[grouped[key]]).sum() for key in date_keys
    ], dtype=np.float64)
    baseline_loss_by_date = np.asarray([
        np.abs(baseline[grouped[key]] - actual[grouped[key]]).sum() for key in date_keys
    ], dtype=np.float64)
    windows_by_date = np.asarray([len(grouped[key]) for key in date_keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled_dates = rng.integers(0, len(date_keys), size=(BOOTSTRAP_DRAWS, len(date_keys)))
    sampled_windows = windows_by_date[sampled_dates].sum(axis=1)
    draws = (
        candidate_loss_by_date[sampled_dates].sum(axis=1)
        - baseline_loss_by_date[sampled_dates].sum(axis=1)
    ) / sampled_windows
    return {
        "estimand": "MAE(candidate)-MAE(reference)",
        "negativeDifferenceFavorsCandidate": True,
        "positiveDifferenceFavorsReference": True,
        "dateBlockCount": len(date_keys),
        "bootstrapDraws": BOOTSTRAP_DRAWS,
        "seed": seed,
        "observedDifferenceMm": mae(actual, candidate) - mae(actual, baseline),
        "confidenceInterval95Mm": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
    }


def evaluate_split(split_name: str, config: dict, horizon: int) -> tuple[dict, list[dict]]:
    rows = read_csv(ROOT / config["rows"])
    by_scenario: dict[str, list[dict[str, str]]] = defaultdict(list)
    date_by_scenario: dict[str, str] = {}
    for row in rows:
        by_scenario[row["scenario_id"]].append(row)
        date_by_scenario[row["scenario_id"]] = row["source_date"]
    for scenario_rows in by_scenario.values():
        scenario_rows.sort(key=lambda row: int(row["simulation_hour"]))
    row_lookup = {
        (scenario_id, int(row["simulation_hour"])): row
        for scenario_id, scenario_rows in by_scenario.items()
        for row in scenario_rows
    }

    prediction_sets = {
        label: read_csv(ROOT / config[label].format(h=horizon))
        for label in ["lstm", "in_sample_hybrid", "oof_hybrid"]
    }
    keys = [
        (row["scenario_id"], int(row["origin_hour"]), int(row["target_hour"]))
        for row in prediction_sets["lstm"]
    ]
    key_set = set(keys)
    if len(key_set) != len(keys):
        raise ValueError(f"Duplicate forecast keys in {split_name}, {horizon} h")
    for label, predictions in prediction_sets.items():
        prediction_keys = {
            (row["scenario_id"], int(row["origin_hour"]), int(row["target_hour"]))
            for row in predictions
        }
        if prediction_keys != key_set:
            raise ValueError(f"Forecast windows differ for {label}, {split_name}, {horizon} h")

    pred_maps = {
        label: {
            (row["scenario_id"], int(row["origin_hour"]), int(row["target_hour"])): row
            for row in predictions
        }
        for label, predictions in prediction_sets.items()
    }
    actual_values: list[float] = []
    baseline_values: list[float] = []
    dates: list[str] = []
    prediction_rows: list[dict] = []
    for scenario_id, origin, target in keys:
        history = [row_lookup[(scenario_id, hour)] for hour in range(origin - LOOKBACK + 1, origin + 1)]
        target_row = row_lookup[(scenario_id, target)]
        rainfall = np.asarray([float(row["cumulative_rainfall_mm"]) for row in history], dtype=np.float64)
        displacement = np.asarray(
            [float(row["rainfall_induced_max_displacement_mm"]) for row in history],
            dtype=np.float64,
        )
        denominator = float(np.dot(rainfall, rainfall))
        slope = float(np.dot(rainfall, displacement) / denominator) if denominator > 1e-18 else 0.0
        current = float(history[-1]["rainfall_induced_max_displacement_mm"])
        target_cumulative_rain = float(target_row["cumulative_rainfall_mm"])
        forecast = max(current, slope * target_cumulative_rain)
        actual = float(pred_maps["lstm"][(scenario_id, origin, target)]["actual_displacement_mm"])
        actual_values.append(actual)
        baseline_values.append(forecast)
        dates.append(date_by_scenario[scenario_id])
        prediction_rows.append({
            "scenario_id": scenario_id,
            "source_date": date_by_scenario[scenario_id],
            "origin_hour": origin,
            "target_hour": target,
            "actual_displacement_mm": actual,
            "proportional_rainfall_prediction_mm": forecast,
            "estimated_slope_mm_per_mm": slope,
            "current_cumulative_rainfall_mm": rainfall[-1],
            "target_cumulative_rainfall_mm": target_cumulative_rain,
            "history_count": LOOKBACK,
        })

    actual = np.asarray(actual_values, dtype=np.float64)
    baseline = np.asarray(baseline_values, dtype=np.float64)
    date_array = np.asarray(dates)
    model_results = {}
    lstm_prediction = np.asarray([
        float(pred_maps["lstm"][key]["lstm_prediction_mm"]) for key in keys
    ], dtype=np.float64)
    seed = BOOTSTRAP_SEED + horizon + (0 if split_name == "primary" else 100 if split_name == "backtest-2024" else 200)
    proportional_vs_lstm = paired_date_bootstrap(actual, baseline, lstm_prediction, date_array, seed)
    for label in ["lstm", "in_sample_hybrid", "oof_hybrid"]:
        predicted = np.asarray([
            float(pred_maps[label][key]["lstm_prediction_mm"])
            if label == "lstm"
            else float(pred_maps[label][key]["physics_guided_prediction_mm"])
            for key in keys
        ], dtype=np.float64)
        model_results[label] = {
            "maeMm": mae(actual, predicted),
            "pairedBootstrapVsProportionalMm": paired_date_bootstrap(
                actual, predicted, baseline, date_array,
                seed,
            ),
        }
        if label != "lstm":
            model_results[label]["pairedBootstrapVsLstmMm"] = paired_date_bootstrap(
                actual, predicted, lstm_prediction, date_array, seed
            )
    return {
        "split": split_name,
        "horizonHours": horizon,
        "windowCount": len(actual),
        "rainfallDateCount": len(set(dates)),
        "proportionalBaseline": {
            "method": "OLS slope through origin on six history pairs (cumulative rainfall, stored displacement); predict slope times target cumulative rainfall; floor at current displacement; no future target used",
            "maeMm": mae(actual, baseline),
            "pairedBootstrapVsLstmMm": proportional_vs_lstm,
            "zeroRainHistoryCount": sum(1 for item in prediction_rows if item["current_cumulative_rainfall_mm"] == 0.0),
            "fallbackForZeroRainHistory": "slope 0; persistence floor",
        },
        "models": model_results,
    }, prediction_rows


def main() -> None:
    results = []
    all_predictions = []
    for split_name, config in SPLITS.items():
        for horizon in (1, 6):
            result, predictions = evaluate_split(split_name, config, horizon)
            results.append(result)
            all_predictions.extend([{**row, "split": split_name, "horizon_hours": horizon} for row in predictions])
    artifact = {
        "id": "TA01-PROPORTIONAL-CUMULATIVE-RAINFALL-BASELINE-V1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "classification": "EXPLORATORY_SEMISYNTHETIC_BASELINE",
        "script": "scripts/evaluate-ta01-proportional-rainfall-baseline.py",
        "dataSources": [
            "data/generated/ta01-fem-500-test.csv",
            "data/generated/ta01-fem-500-backtest-2024-test.csv",
            "data/generated/ta01-fem-500-chronological-test.csv",
            "archived LSTM, in-sample hybrid and OOF prediction CSV files listed in script configuration",
        ],
        "lookbackHours": LOOKBACK,
        "predictionConstraint": "forecast cannot be less than current stored displacement",
        "bootstrap": {"unit": "source_date", "draws": BOOTSTRAP_DRAWS, "seedBase": BOOTSTRAP_SEED},
        "limitations": [
            "Uses the semisynthetic generator's cumulative-rainfall feature and stored target history.",
            "The proportional form is motivated by the current linear-elastic generator and is not assumed valid for field talus.",
            "For a zero-rainfall history, no slope is identifiable and the forecast falls back to persistence.",
            "A stored zero can reflect output quantization rather than exactly zero physical displacement.",
        ],
        "results": results,
    }
    output_json = ROOT / "data" / "validation" / "ta01-proportional-rainfall-baseline.json"
    output_csv = ROOT / "data" / "validation" / "ta01-proportional-rainfall-baseline-predictions.csv"
    output_json.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_predictions[0].keys()))
        writer.writeheader()
        writer.writerows(all_predictions)
    print(json.dumps({"json": str(output_json.relative_to(ROOT)), "csv": str(output_csv.relative_to(ROOT)), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
