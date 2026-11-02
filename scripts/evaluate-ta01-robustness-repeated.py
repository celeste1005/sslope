#!/usr/bin/env python3
"""Repeat stochastic TA-01 test perturbations and compare fixed correctors."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DRAWS = 30
DRAW_SEED_BASE = 20261060
BOOTSTRAP_DRAWS = 20_000
OOF_CORRECTOR_SEEDS = (20261030, 20261031, 20261032)
FREE_MLP_FILE_SUFFIXES = (20261029, 20261031, 20261032)
CONDITIONS = (
    ("NOISE_1PCT_STD", "gaussian", 0.01),
    ("NOISE_5PCT_STD", "gaussian", 0.05),
    ("MISSING_10PCT_MCAR", "missing", 0.10),
    ("MISSING_30PCT_MCAR", "missing", 0.30),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model_artifacts(horizon: int) -> dict[str, list[tuple[str, dict]]]:
    models = {
        "in_sample_hybrid": [("seed20260917", read_json(ROOT / f"data/models/ta01-physics-guided-{horizon}h.json"))],
        "oof_hybrid": [(f"corrector_seed{seed}", read_json(ROOT / f"data/models/ta01-physics-guided-{horizon}h-oof.json" if seed == 20261030 else ROOT / f"data/models/ta01-primary-physics-guided-{horizon}h-oof-s{seed}.json")) for seed in OOF_CORRECTOR_SEEDS],
        "free_mlp_oof": [(f"file_suffix{file_suffix}", read_json(ROOT / f"data/models/ta01-free-corrector-{horizon}h-s{file_suffix}.json")) for file_suffix in FREE_MLP_FILE_SUFFIXES],
    }
    return models


def make_model(artifact: dict, physics):
    input_size = len(physics.CORRECTOR_FEATURES)
    hidden = int(artifact["architecture"]["hiddenUnits"])
    seed = int(artifact["training"]["seed"])
    if artifact["architecture"].get("correctionMode") == "free":
        model = physics.FreeResidualCorrector(input_size, hidden, seed)
    else:
        model = physics.PhysicsCorrector(
            input_size,
            hidden,
            seed,
            physics.CORRECTOR_FEATURES.index("forecast_rainfall_mm"),
            physics.CORRECTOR_FEATURES.index("mohr_coulomb_safety_index"),
        )
    model.parameters = {name: np.asarray(value, dtype=np.float64) for name, value in artifact["model"]["weights"].items()}
    mean = np.asarray([artifact["model"]["inputMean"][name] for name in physics.CORRECTOR_FEATURES], dtype=np.float64)
    std = np.asarray([artifact["model"]["inputStandardDeviation"][name] for name in physics.CORRECTOR_FEATURES], dtype=np.float64)
    return model, mean, std


def lstm_model(horizon: int, lstm):
    artifact = read_json(ROOT / f"data/models/ta01-lstm-{horizon}h.json")
    network = lstm.LstmRegressor(len(lstm.FEATURES), artifact["architecture"]["hiddenUnits"], artifact["training"]["seed"])
    network.parameters = {name: np.asarray(value, dtype=np.float64) for name, value in artifact["model"]["weights"].items()}
    mean = np.asarray([artifact["model"]["featureMean"][name] for name in lstm.FEATURES], dtype=np.float64).reshape(1, 1, -1)
    std = np.asarray([artifact["model"]["featureStandardDeviation"][name] for name in lstm.FEATURES], dtype=np.float64).reshape(1, 1, -1)
    return artifact, network, mean, std


def predict(dataset, x, lstm_artifact, lstm_network, lstm_mean, lstm_std, corrector, corrector_mean, corrector_std, physics):
    standardized_delta = lstm_network.forward((x - lstm_mean) / lstm_std)
    delta_mean = float(lstm_artifact["model"]["targetDeltaMean"])
    delta_std = float(lstm_artifact["model"]["targetDeltaStandardDeviation"])
    lstm_pred = np.maximum(dataset.current, dataset.current + delta_mean + standardized_delta * delta_std)
    result = {"lstm": lstm_pred}
    if corrector is not None:
        # This recomputes the LSTM delta and all three last-minus-first history
        # differences from the perturbed sequence for each draw.
        corrector_x = physics.corrector_inputs_from_sequence(x, standardized_delta)
        corrected_delta = standardized_delta + corrector.forward((corrector_x - corrector_mean) / corrector_std)
        result["corrected"] = np.maximum(dataset.current, dataset.current + delta_mean + corrected_delta * delta_std)
    return result


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_DRAWS, values.size))
    means = values[indices].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def summarize_draws(draw_metrics: list[dict], clean_mae: float, seed: int) -> dict:
    maes = np.asarray([row["maeMm"] for row in draw_metrics], dtype=np.float64)
    increases = 100.0 * (maes / clean_mae - 1.0)
    mae_ci = bootstrap_mean_ci(maes, seed)
    increase_ci = bootstrap_mean_ci(increases, seed + 1)
    return {
        "drawCount": int(maes.size),
        "meanMaeMm": float(maes.mean()),
        "sdAcrossDrawsMm": float(maes.std(ddof=1)),
        "bootstrapMean95Mm": list(mae_ci),
        "meanIncreasePercentVsSameModelClean": float(increases.mean()),
        "bootstrapMeanIncrease95Percent": list(increase_ci),
        "cleanMaeMm": float(clean_mae),
    }


def evaluate_horizon(horizon: int, lstm, physics) -> tuple[dict, list[dict]]:
    test_rows = lstm.read_csv(ROOT / "data/generated/ta01-fem-500-test.csv")
    dataset = lstm.make_sequences(test_rows, 6, horizon)
    lstm_artifact, lstm_network, lstm_mean, lstm_std = lstm_model(horizon, lstm)
    feature_std = lstm_std.reshape(1, 1, -1)
    feature_mean = lstm_mean.reshape(1, 1, -1)
    model_files = model_artifacts(horizon)
    models: dict[str, list[dict]] = {"lstm": []}
    for variant, artifact_pairs in model_files.items():
        models[variant] = []
        for label, artifact in artifact_pairs:
            corrector, c_mean, c_std = make_model(artifact, physics)
            actual_seed = int(artifact["training"]["seed"])
            models[variant].append({"label": f"corrector_seed{actual_seed}", "artifact": artifact, "corrector": corrector, "mean": c_mean, "std": c_std})

    clean_predictions: dict[str, list[np.ndarray]] = {name: [] for name in models}
    for name, entries in models.items():
        if name == "lstm":
            pred = predict(dataset, dataset.x, lstm_artifact, lstm_network, lstm_mean, lstm_std, None, None, None, physics)["lstm"]
            clean_predictions[name].append(pred)
        else:
            for entry in entries:
                pred = predict(dataset, dataset.x, lstm_artifact, lstm_network, lstm_mean, lstm_std, entry["corrector"], entry["mean"], entry["std"], physics)["corrected"]
                clean_predictions[name].append(pred)
    clean_mae_by_model = {
        name: float(np.mean([np.mean(np.abs(pred - dataset.actual)) for pred in predictions]))
        for name, predictions in clean_predictions.items()
    }

    date_by_scenario = {str(row["scenario_id"]): str(row["source_date"]) for row in test_rows}
    date_order = sorted({date_by_scenario[str(row["scenario_id"])] for row in dataset.metadata})
    date_indices = {date: [] for date in date_order}
    for idx, row in enumerate(dataset.metadata):
        date_indices[date_by_scenario[str(row["scenario_id"])]].append(idx)

    result_rows: list[dict] = []
    per_condition: list[dict] = []
    for condition_index, (condition, kind, level) in enumerate(CONDITIONS):
        draw_results = {name: [] for name in models}
        rng_seeds = [DRAW_SEED_BASE + horizon * 1000 + condition_index * 100 + draw for draw in range(DRAWS)]
        for draw_idx, seed in enumerate(rng_seeds):
            rng = np.random.default_rng(seed)
            if kind == "gaussian":
                perturbed = dataset.x + rng.normal(0.0, level, size=dataset.x.shape) * feature_std
            else:
                perturbed = dataset.x.copy()
                mask = rng.random(perturbed.shape) < level
                replacement = np.broadcast_to(feature_mean, perturbed.shape)
                perturbed[mask] = replacement[mask]

            for model_name in models:
                entries = models[model_name]
                if model_name == "lstm":
                    pred = predict(dataset, perturbed, lstm_artifact, lstm_network, lstm_mean, lstm_std, None, None, None, physics)["lstm"]
                    predictions = [pred]
                else:
                    predictions = [predict(dataset, perturbed, lstm_artifact, lstm_network, lstm_mean, lstm_std, entry["corrector"], entry["mean"], entry["std"], physics)["corrected"] for entry in entries]
                per_fit = []
                for fit_idx, pred in enumerate(predictions):
                    error = pred - dataset.actual
                    mae = float(np.mean(np.abs(error)))
                    rmse = float(np.sqrt(np.mean(error ** 2)))
                    per_fit.append({"fit": entries[fit_idx]["label"] if model_name != "lstm" else lstm_artifact["id"], "maeMm": mae, "rmseMm": rmse})
                draw_mean_mae = float(np.mean([item["maeMm"] for item in per_fit]))
                draw_mean_rmse = float(np.mean([item["rmseMm"] for item in per_fit]))
                draw_results[model_name].append({"drawIndex": draw_idx + 1, "seed": seed, "maeMm": draw_mean_mae, "rmseMm": draw_mean_rmse, "perFit": per_fit})

        summarized = {}
        for model_idx, (model_name, draws) in enumerate(draw_results.items()):
            summary = summarize_draws(draws, clean_mae_by_model[model_name], DRAW_SEED_BASE + horizon * 10_000 + condition_index * 100 + model_idx * 2)
            summary["fitCountAveragedPerDraw"] = len(clean_predictions[model_name])
            summary["cleanMaePerFitMm"] = [float(np.mean(np.abs(pred - dataset.actual))) for pred in clean_predictions[model_name]]
            summary["perFitLabel"] = [entry["label"] for entry in models[model_name]] if model_name != "lstm" else [lstm_artifact["id"]]
            summarized[model_name] = summary
            for draw in draws:
                result_rows.append({
                    "horizonHours": horizon,
                    "condition": condition,
                    "drawIndex": draw["drawIndex"],
                    "drawSeed": draw["seed"],
                    "model": model_name,
                    "meanAcrossFitsMaeMm": draw["maeMm"],
                    "meanAcrossFitsRmseMm": draw["rmseMm"],
                    "fitCount": len(draw["perFit"]),
                    "perFit": json.dumps(draw["perFit"], ensure_ascii=False),
                })
        per_condition.append({
            "condition": condition,
            "kind": kind,
            "level": level,
            "windowCount": int(dataset.actual.size),
            "dateCount": len(date_order),
            "dateBlockCounts": {"min": min(map(len, date_indices.values())), "max": max(map(len, date_indices.values()))},
            "summary": summarized,
            "draws": {name: [{k: v for k, v in d.items() if k != "perFit"} | {"perFit": d["perFit"]} for d in values] for name, values in draw_results.items()},
        })

    files = [
        ROOT / "data/generated/ta01-fem-500-test.csv",
        ROOT / f"data/models/ta01-lstm-{horizon}h.json",
        ROOT / f"data/models/ta01-physics-guided-{horizon}h.json",
        ROOT / f"data/models/ta01-physics-guided-{horizon}h-oof.json",
        ROOT / f"data/models/ta01-primary-physics-guided-{horizon}h-oof-s20261031.json",
        ROOT / f"data/models/ta01-primary-physics-guided-{horizon}h-oof-s20261032.json",
        *(ROOT / f"data/models/ta01-free-corrector-{horizon}h-s{file_suffix}.json" for file_suffix in FREE_MLP_FILE_SUFFIXES),
    ]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(files))}
    return {
        "horizonHours": horizon,
        "testLstmId": lstm_artifact["id"],
        "modelCorrectors": {name: [entry["artifact"]["id"] for entry in entries] for name, entries in models.items() if name != "lstm"},
        "fixedTestDates": len(date_order),
        "windowCount": int(dataset.actual.size),
        "cleanMaeMmByModel": clean_mae_by_model,
        "perturbationDraws": DRAWS,
        "conditions": per_condition,
        "sourceSha256": hashes,
    }, result_rows


def main() -> None:
    lstm = load_module("ta01_robust_repeated_lstm", ROOT / "scripts/train-lstm.py")
    physics = load_module("ta01_robust_repeated_physics", ROOT / "scripts/train-physics-guided.py")
    all_results, all_rows = [], []
    for horizon in (1, 6):
        summary, rows = evaluate_horizon(horizon, lstm, physics)
        all_results.append(summary)
        all_rows.extend(rows)
    output_json = ROOT / "data/validation/ta01-robustness-repeated.json"
    output_csv = ROOT / "data/validation/ta01-robustness-repeated-draws.csv"
    artifact = {
        "id": "TA01-ROBUSTNESS-REPEATED-30-DRAWS-V2",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "classification": "EXPLORATORY_FIXED_SEMIsynthetic_TEST_PERTURBATION_REPLICATIONS",
        "method": "Thirty independent Gaussian-noise or MCAR mean-imputation perturbation draws per condition and horizon. All models receive identical corrupted input in each draw. MAE/RMSE intervals are percentile bootstrap intervals over perturbation draws only, conditional on the fixed test dates and fixed model fits. Corrector-derived inputs are recomputed from each corrupted sequence.",
        "perturbationDrawCount": DRAWS,
        "drawSeedBase": DRAW_SEED_BASE,
        "bootstrapDraws": BOOTSTRAP_DRAWS,
        "bootstrapUnit": "perturbation_draw; test dates remain fixed",
        "limitations": [
            "The inference models and test dates are fixed; intervals quantify variation across stochastic corruption draws, not field generalization or model-initialization uncertainty.",
            "The three OOF and three free-MLP fits are summarized by averaging their per-fit MAE within each corruption draw; per-fit values are retained in the draw CSV.",
            "The clean current-displacement floor is retained, consistent with the previous robustness protocol.",
            "Noise and missingness act independently on all 15 input channels and six sequence steps; this is a stress test and not a realistic sensor-failure process.",
            "The held-out target remains unperturbed and is used only for scoring.",
        ],
        "results": all_results,
    }
    output_json.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = ["horizonHours", "condition", "drawIndex", "drawSeed", "model", "meanAcrossFitsMaeMm", "meanAcrossFitsRmseMm", "fitCount", "perFit"]
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps({"json": output_json.relative_to(ROOT).as_posix(), "csv": output_csv.relative_to(ROOT).as_posix(), "horizons": [{"horizonHours": r["horizonHours"], "models": list(r["conditions"][0]["summary"].keys())} for r in all_results]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
