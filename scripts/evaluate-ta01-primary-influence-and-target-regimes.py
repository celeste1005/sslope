#!/usr/bin/env python3
"""Audit primary-split date influence and stored-zero regimes for TA-01.

Reports the full-sample comparison first, then explicit sensitivity analyses
including the previously identified extreme-rainfall scenario. No observation
is removed from the primary estimate.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20261050
EXTREME_SCENARIO = "TA01-S0001"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def bootstrap_date_difference(groups: dict[str, tuple[float, float, int]], seed: int) -> dict:
    dates = sorted(groups)
    lstm_sum = np.asarray([groups[d][0] for d in dates], dtype=np.float64)
    hybrid_sum = np.asarray([groups[d][1] for d in dates], dtype=np.float64)
    windows = np.asarray([groups[d][2] for d in dates], dtype=np.float64)
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(dates), size=(BOOTSTRAP_DRAWS, len(dates)))
    n = windows[choices].sum(axis=1)
    draws = (lstm_sum[choices].sum(axis=1) - hybrid_sum[choices].sum(axis=1)) / n
    full_lstm = float(lstm_sum.sum() / windows.sum())
    full_hybrid = float(hybrid_sum.sum() / windows.sum())
    return {
        "estimand": "MAE_LSTM-MAE_in_sample_hybrid; positive favors hybrid",
        "dateBlockCount": len(dates),
        "bootstrapDraws": BOOTSTRAP_DRAWS,
        "seed": seed,
        "lstmMaeMm": full_lstm,
        "hybridMaeMm": full_hybrid,
        "observedDifferenceMm": full_lstm - full_hybrid,
        "confidenceInterval95Mm": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
    }


def summarize_horizon(horizon: int, rows: list[dict[str, str]], predictions: list[dict[str, str]]) -> dict:
    metadata = {row["scenario_id"]: row for row in rows}
    by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in predictions:
        scenario = row["scenario_id"]
        by_date[metadata[scenario]["source_date"]].append(row)

    grouped_losses: dict[str, tuple[float, float, int]] = {}
    per_date = []
    for date, date_rows in sorted(by_date.items()):
        lstm_loss = sum(abs(float(r["actual_displacement_mm"]) - float(r["lstm_prediction_mm"])) for r in date_rows)
        hybrid_loss = sum(abs(float(r["actual_displacement_mm"]) - float(r["physics_guided_prediction_mm"])) for r in date_rows)
        grouped_losses[date] = (lstm_loss, hybrid_loss, len(date_rows))
        scenarios = sorted({r["scenario_id"] for r in date_rows})
        per_date.append({
            "sourceDate": date,
            "scenarioIds": scenarios,
            "observedDailyRainfallMm": [float(metadata[s]["observed_daily_rainfall_mm"]) for s in scenarios],
            "windowCount": len(date_rows),
            "lstmAbsoluteErrorSumMm": lstm_loss,
            "hybridAbsoluteErrorSumMm": hybrid_loss,
            "lstmMinusHybridMaeContributionMm": (lstm_loss - hybrid_loss) / len(date_rows),
        })
    paired = bootstrap_date_difference(grouped_losses, BOOTSTRAP_SEED + horizon)

    full_rows = predictions
    dry = [r for r in full_rows if float(metadata[r["scenario_id"]]["observed_daily_rainfall_mm"]) == 0.0]
    wet = [r for r in full_rows if float(metadata[r["scenario_id"]]["observed_daily_rainfall_mm"]) > 0.0]
    target_regimes = {}
    for name, subset in [("dailyRainfallZero", dry), ("dailyRainfallPositive", wet)]:
        target_zero = sum(float(r["actual_displacement_mm"]) == 0.0 for r in subset)
        target_regimes[name] = {
            "sampleCount": len(subset),
            "zeroTargetCount": target_zero,
            "zeroTargetShare": target_zero / len(subset) if subset else None,
            "lstmMaeMm": sum(abs(float(r["actual_displacement_mm"]) - float(r["lstm_prediction_mm"])) for r in subset) / len(subset) if subset else None,
            "hybridMaeMm": sum(abs(float(r["actual_displacement_mm"]) - float(r["physics_guided_prediction_mm"])) for r in subset) / len(subset) if subset else None,
        }

    scenario_sums: dict[str, dict[str, float | int]] = defaultdict(lambda: {"lstm": 0.0, "hybrid": 0.0, "count": 0})
    for r in predictions:
        key = r["scenario_id"]
        scenario_sums[key]["lstm"] += abs(float(r["actual_displacement_mm"]) - float(r["lstm_prediction_mm"]))
        scenario_sums[key]["hybrid"] += abs(float(r["actual_displacement_mm"]) - float(r["physics_guided_prediction_mm"]))
        scenario_sums[key]["count"] += 1
    n_all = len(predictions)
    extreme = scenario_sums[EXTREME_SCENARIO]
    without_extreme_count = n_all - int(extreme["count"])
    influence = {
        "scenarioId": EXTREME_SCENARIO,
        "sourceDate": metadata[EXTREME_SCENARIO]["source_date"],
        "dailyRainfallMm": float(metadata[EXTREME_SCENARIO]["observed_daily_rainfall_mm"]),
        "windowCount": int(extreme["count"]),
        "lstmAbsoluteErrorShare": float(extreme["lstm"] / sum(v["lstm"] for v in scenario_sums.values())),
        "hybridAbsoluteErrorShare": float(extreme["hybrid"] / sum(v["hybrid"] for v in scenario_sums.values())),
        "fullSampleMaeMm": {"lstm": paired["lstmMaeMm"], "inSampleHybrid": paired["hybridMaeMm"]},
        "excludingScenarioMaeMm": {
            "lstm": float((sum(v["lstm"] for v in scenario_sums.values()) - float(extreme["lstm"])) / without_extreme_count),
            "inSampleHybrid": float((sum(v["hybrid"] for v in scenario_sums.values()) - float(extreme["hybrid"])) / without_extreme_count),
        },
        "interpretation": "Sensitivity only; the full test set remains the primary estimate.",
    }
    per_date.sort(key=lambda x: abs(x["lstmMinusHybridMaeContributionMm"]), reverse=True)
    return {
        "horizonHours": horizon,
        "windowCount": n_all,
        "rainfallDateCount": len(by_date),
        "pairedDateBootstrap": paired,
        "targetZeroByObservedDailyRainfall": target_regimes,
        "extremeScenarioSensitivity": influence,
        "largestAbsoluteDateContributions": per_date[:10],
        "allDateContributions": per_date,
    }


def main() -> None:
    rows = read_csv(ROOT / "data/generated/ta01-fem-500-test.csv")
    results = []
    for horizon in (1, 6):
        predictions = read_csv(ROOT / f"data/generated/ta01-physics-guided-{horizon}h-test-predictions.csv")
        results.append(summarize_horizon(horizon, rows, predictions))
    artifact = {
        "id": "TA01-PRIMARY-INFLUENCE-AND-TARGET-REGIMES-V1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "classification": "EXPLORATORY_AUDIT_EXISTING_PRIMARY_ARTIFACTS",
        "script": "scripts/evaluate-ta01-primary-influence-and-target-regimes.py",
        "inputRows": "data/generated/ta01-fem-500-test.csv",
        "predictionFiles": ["data/generated/ta01-physics-guided-1h-test-predictions.csv", "data/generated/ta01-physics-guided-6h-test-predictions.csv"],
        "primaryAnalysisUsesAllDates": True,
        "extremeScenarioExclusionIsSensitivityOnly": True,
        "bootstrap": {"unit": "source_date", "draws": BOOTSTRAP_DRAWS, "seedBase": BOOTSTRAP_SEED},
        "results": results,
    }
    out = ROOT / "data/validation/ta01-primary-influence-and-target-regimes.json"
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out.relative_to(ROOT)), "results": [
        {"horizon": r["horizonHours"], "pairedDateBootstrap": r["pairedDateBootstrap"], "targetZeroByObservedDailyRainfall": r["targetZeroByObservedDailyRainfall"], "extremeScenarioSensitivity": r["extremeScenarioSensitivity"]}
        for r in results
    ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
