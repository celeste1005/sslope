#!/usr/bin/env python3
"""Generate date-grouped out-of-fold LSTM predictions for TA-01 training windows."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_lstm_module(project_dir: Path):
    path = project_dir / "scripts" / "train-lstm.py"
    spec = importlib.util.spec_from_file_location("ta01_train_lstm_oof", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fit_fold(module, train_data, validation_data, seed: int, epochs: int,
             patience: int, batch_size: int, learning_rate: float):
    feature_mean = train_data.x.mean(axis=(0, 1), keepdims=True)
    feature_std = train_data.x.std(axis=(0, 1), keepdims=True)
    feature_std[feature_std < 1e-12] = 1.0
    target_mean = float(train_data.target_delta.mean())
    target_std = float(train_data.target_delta.std()) or 1.0
    train_x = (train_data.x - feature_mean) / feature_std
    train_y = (train_data.target_delta - target_mean) / target_std

    model = module.LstmRegressor(len(module.FEATURES), 12, seed)
    optimizer = module.Adam(model.parameters, learning_rate)
    random = np.random.default_rng(seed)
    best_parameters = copy.deepcopy(model.parameters)
    best_validation_mae = math.inf
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        order = random.permutation(train_x.shape[0])
        for start in range(0, order.size, batch_size):
            indices = order[start:start + batch_size]
            prediction, cache, final_hidden = model.forward(train_x[indices], keep_cache=True)
            gradients = model.backward(prediction - train_y[indices], cache, final_hidden)
            optimizer.step(model.parameters, gradients)

        _, validation_prediction = module.predict_physical(
            model, validation_data, feature_mean, feature_std, target_mean, target_std
        )
        validation_mae = float(np.mean(np.abs(validation_prediction - validation_data.actual)))
        if validation_mae < best_validation_mae - 1e-10:
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_parameters = copy.deepcopy(model.parameters)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.parameters = best_parameters
    return model, feature_mean, feature_std, target_mean, target_std, best_epoch, best_validation_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=[1, 6], required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--input-prefix", default="data/generated/ta01-fem-500")
    parser.add_argument("--lstm-model", default=None,
                        help="Modelo final LSTM usado en validación y prueba; define la escala objetivo")
    parser.add_argument("--output", default=None)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("Se requieren al menos dos folds")

    project_dir = Path(__file__).resolve().parent.parent
    module = load_lstm_module(project_dir)
    prefix = project_dir / args.input_prefix
    train_rows = module.read_csv(Path(f"{prefix}-train.csv"))
    validation_rows = module.read_csv(Path(f"{prefix}-validation.csv"))
    train_data = module.make_sequences(train_rows, 6, args.horizon)
    validation_data = module.make_sequences(validation_rows, 6, args.horizon)

    lstm_path = project_dir / (args.lstm_model or f"data/models/ta01-lstm-{args.horizon}h.json")
    final_artifact = json.loads(lstm_path.read_text(encoding="utf-8"))
    final_target_mean = float(final_artifact["model"]["targetDeltaMean"])
    final_target_std = float(final_artifact["model"]["targetDeltaStandardDeviation"])

    date_by_scenario = {row["scenario_id"]: row["source_date"] for row in train_rows}
    scenario_dates = sorted({date_by_scenario[item["scenario_id"]] for item in train_data.metadata})
    rng = np.random.default_rng(args.seed)
    shuffled_dates = list(np.asarray(scenario_dates)[rng.permutation(len(scenario_dates))])
    date_folds = [set(shuffled_dates[index::args.folds]) for index in range(args.folds)]

    prediction_by_key = {}
    fold_summaries = []
    for fold_index, held_dates in enumerate(date_folds):
        held_scenarios = {
            scenario_id for scenario_id, source_date in date_by_scenario.items()
            if source_date in held_dates
        }
        fold_train_rows = [row for row in train_rows if row["scenario_id"] not in held_scenarios]
        fold_held_rows = [row for row in train_rows if row["scenario_id"] in held_scenarios]
        fold_train_data = module.make_sequences(fold_train_rows, 6, args.horizon)
        fold_held_data = module.make_sequences(fold_held_rows, 6, args.horizon)
        fold_seed = args.seed + fold_index + 1
        model, feature_mean, feature_std, target_mean, target_std, best_epoch, val_mae = fit_fold(
            module, fold_train_data, validation_data, fold_seed, args.epochs,
            args.patience, args.batch_size, args.learning_rate,
        )
        standardized_x = (fold_held_data.x - feature_mean) / feature_std
        fold_standardized_delta = model.forward(standardized_x)
        physical_delta = target_mean + fold_standardized_delta * target_std
        final_standardized_delta = (physical_delta - final_target_mean) / final_target_std
        for metadata, predicted_delta in zip(fold_held_data.metadata, final_standardized_delta):
            key = (metadata["scenario_id"], int(metadata["origin_hour"]), int(metadata["target_hour"]))
            if key in prediction_by_key:
                raise RuntimeError(f"Ventana duplicada entre folds: {key}")
            prediction_by_key[key] = (float(predicted_delta), fold_index + 1)
        fold_summaries.append({
            "fold": fold_index + 1,
            "heldSourceDateCount": len(held_dates),
            "heldScenarioCount": len(held_scenarios),
            "heldWindowCount": len(fold_held_data.metadata),
            "trainingScenarioCount": len({row["scenario_id"] for row in fold_train_rows}),
            "selectedEpoch": best_epoch,
            "validationMaeMm": val_mae,
            "seed": fold_seed,
        })
        print(f"fold {fold_index + 1}/{args.folds}: held dates={len(held_dates)}, windows={len(fold_held_data.metadata)}, best_epoch={best_epoch}, validation_mae_mm={val_mae:.10g}", flush=True)

    expected_keys = {
        (item["scenario_id"], int(item["origin_hour"]), int(item["target_hour"]))
        for item in train_data.metadata
    }
    if set(prediction_by_key) != expected_keys:
        raise RuntimeError(f"Cobertura OOF inválida: {len(prediction_by_key)} predicciones para {len(expected_keys)} ventanas")

    output_path = project_dir / (args.output or f"data/validation/ta01-lstm-{args.horizon}h-oof-train-predictions.csv")
    manifest_path = project_dir / (args.manifest or f"data/validation/ta01-lstm-{args.horizon}h-oof-manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["scenario_id", "origin_hour", "target_hour", "source_date", "fold", "lstm_delta_standardized"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for metadata in train_data.metadata:
            key = (metadata["scenario_id"], int(metadata["origin_hour"]), int(metadata["target_hour"]))
            prediction, fold = prediction_by_key[key]
            writer.writerow({
                **metadata,
                "source_date": date_by_scenario[metadata["scenario_id"]],
                "fold": fold,
                "lstm_delta_standardized": f"{prediction:.12g}",
            })

    manifest = {
        "method": "DATE_GROUPED_K_FOLD_OUT_OF_FOLD_LSTM_PREDICTIONS",
        "horizonHours": args.horizon,
        "lookbackHours": 6,
        "folds": args.folds,
        "foldAssignmentSeed": args.seed,
        "baseLstmModel": str(lstm_path.relative_to(project_dir)),
        "trainingInputPrefix": args.input_prefix,
        "validationSplitUsedOnlyForFoldEarlyStopping": True,
        "dateGroupLeakageControl": "All scenarios with the same source_date stay in one held-out fold.",
        "trainingWindows": len(train_data.metadata),
        "uniqueTrainingDates": len(scenario_dates),
        "foldSummaries": fold_summaries,
        "predictionFile": str(output_path.relative_to(project_dir)),
        "correctorPredictionScale": "Standardized physical displacement increment using the final LSTM artifact's training target mean and standard deviation.",
        "limitations": [
            "The original validation split selects early stopping for each fold; it is not used to fit the corrector.",
            "This is cross-fitting for the primary date-grouped split, not a multi-seed reliability analysis.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"predictions": str(output_path), "manifest": str(manifest_path), "trainingWindows": len(prediction_by_key)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
