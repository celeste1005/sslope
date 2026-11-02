#!/usr/bin/env python3
"""Entrena un corrector neuronal físico sobre la LSTM temporal de TA-01."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CORRECTOR_FEATURES = [
    "lstm_delta_standardized",
    "current_displacement_mm",
    "current_rainfall_mm_h",
    "cumulative_rainfall_mm",
    "forecast_rainfall_mm",
    "maximum_pore_pressure_kpa",
    "mohr_coulomb_safety_index",
    "simulation_hour",
    "cohesion_kpa",
    "friction_angle_deg",
    "unit_weight_kn_m3",
    "young_modulus_mpa",
    "log10_permeability_m_s",
    "drainage_efficiency",
    "water_table_m",
    "storage_coefficient",
    "lookback_displacement_change_mm",
    "lookback_pore_pressure_change_kpa",
    "lookback_safety_index_change",
]


def load_lstm_module(project_dir: Path):
    path = project_dir / "scripts" / "train-lstm.py"
    spec = importlib.util.spec_from_file_location("ta01_train_lstm", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def serializable_array(value: np.ndarray):
    return np.round(value, 12).tolist()


class PhysicsCorrector:
    def __init__(self, input_size: int, hidden_size: int, seed: int, rain_index: int, safety_index: int):
        random = np.random.default_rng(seed)
        scale = math.sqrt(2.0 / (input_size + hidden_size))
        self.rain_index = rain_index
        self.safety_index = safety_index
        self.parameters = {
            "w1": random.normal(0.0, scale, (input_size, hidden_size)),
            "b1": np.zeros(hidden_size, dtype=np.float64),
            "w2": np.zeros((hidden_size, 1), dtype=np.float64),
            "b2": np.zeros(1, dtype=np.float64),
            "rainRaw": np.asarray([-4.0], dtype=np.float64),
            "safetyRaw": np.asarray([-4.0], dtype=np.float64),
        }

    def forward(self, x: np.ndarray, keep_cache: bool = False):
        network_x = x.copy()
        network_x[:, self.rain_index] = 0.0
        network_x[:, self.safety_index] = 0.0
        hidden = np.tanh(network_x @ self.parameters["w1"] + self.parameters["b1"])
        rain_coefficient = np.logaddexp(0.0, self.parameters["rainRaw"])[0]
        safety_coefficient = np.logaddexp(0.0, self.parameters["safetyRaw"])[0]
        correction = (hidden @ self.parameters["w2"] + self.parameters["b2"]).reshape(-1)
        correction += rain_coefficient * x[:, self.rain_index] - safety_coefficient * x[:, self.safety_index]
        return (correction, (x, network_x, hidden)) if keep_cache else correction

    def backward(self, output_gradient: np.ndarray, cache) -> dict[str, np.ndarray]:
        x, network_x, hidden = cache
        column_gradient = output_gradient.reshape(-1, 1)
        hidden_gradient = (column_gradient @ self.parameters["w2"].T) * (1.0 - hidden ** 2)
        rain_sigmoid = 1.0 / (1.0 + np.exp(-self.parameters["rainRaw"][0]))
        safety_sigmoid = 1.0 / (1.0 + np.exp(-self.parameters["safetyRaw"][0]))
        return {
            "w1": network_x.T @ hidden_gradient,
            "b1": hidden_gradient.sum(axis=0),
            "w2": hidden.T @ column_gradient,
            "b2": column_gradient.sum(axis=0),
            "rainRaw": np.asarray([float(np.sum(output_gradient * x[:, self.rain_index])) * rain_sigmoid]),
            "safetyRaw": np.asarray([-float(np.sum(output_gradient * x[:, self.safety_index])) * safety_sigmoid]),
        }


class FreeResidualCorrector:
    """Same-size residual MLP without hard monotonic parameterization."""

    def __init__(self, input_size: int, hidden_size: int, seed: int):
        random = np.random.default_rng(seed)
        scale = math.sqrt(2.0 / (input_size + hidden_size))
        self.parameters = {
            "w1": random.normal(0.0, scale, (input_size, hidden_size)),
            "b1": np.zeros(hidden_size, dtype=np.float64),
            "w2": np.zeros((hidden_size, 1), dtype=np.float64),
            "b2": np.zeros(1, dtype=np.float64),
        }

    def forward(self, x: np.ndarray, keep_cache: bool = False):
        hidden = np.tanh(x @ self.parameters["w1"] + self.parameters["b1"])
        correction = (hidden @ self.parameters["w2"] + self.parameters["b2"]).reshape(-1)
        return (correction, (x, hidden)) if keep_cache else correction

    def backward(self, output_gradient: np.ndarray, cache) -> dict[str, np.ndarray]:
        x, hidden = cache
        column_gradient = output_gradient.reshape(-1, 1)
        hidden_gradient = (column_gradient @ self.parameters["w2"].T) * (1.0 - hidden ** 2)
        return {
            "w1": x.T @ hidden_gradient,
            "b1": hidden_gradient.sum(axis=0),
            "w2": hidden.T @ column_gradient,
            "b2": column_gradient.sum(axis=0),
        }

class Adam:
    def __init__(self, parameters: dict[str, np.ndarray], learning_rate: float):
        self.learning_rate = learning_rate
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.step_number = 0

    def step(self, parameters: dict[str, np.ndarray], gradients: dict[str, np.ndarray]):
        self.step_number += 1
        norm = math.sqrt(sum(float(np.sum(gradient ** 2)) for gradient in gradients.values()))
        if norm > 5.0:
            gradients = {name: gradient * (5.0 / norm) for name, gradient in gradients.items()}
        for name in parameters:
            self.first[name] = 0.9 * self.first[name] + 0.1 * gradients[name]
            self.second[name] = 0.999 * self.second[name] + 0.001 * gradients[name] ** 2
            first_corrected = self.first[name] / (1.0 - 0.9 ** self.step_number)
            second_corrected = self.second[name] / (1.0 - 0.999 ** self.step_number)
            parameters[name] -= self.learning_rate * first_corrected / (np.sqrt(second_corrected) + 1e-8)


def add_gradients(*groups):
    return {name: sum(group[name] for group in groups) for name in groups[0]}


def load_lstm(model_path: Path, lstm_module):
    artifact = json.loads(model_path.read_text(encoding="utf-8"))
    network = lstm_module.LstmRegressor(len(lstm_module.FEATURES), artifact["architecture"]["hiddenUnits"], artifact["training"]["seed"])
    network.parameters = {name: np.asarray(value, dtype=np.float64) for name, value in artifact["model"]["weights"].items()}
    feature_mean = np.asarray([artifact["model"]["featureMean"][name] for name in lstm_module.FEATURES], dtype=np.float64).reshape(1, 1, -1)
    feature_std = np.asarray([artifact["model"]["featureStandardDeviation"][name] for name in lstm_module.FEATURES], dtype=np.float64).reshape(1, 1, -1)
    return artifact, network, feature_mean, feature_std


def lstm_standardized_delta(network, dataset, feature_mean, feature_std):
    return network.forward((dataset.x - feature_mean) / feature_std)


def corrector_inputs_from_sequence(sequence_x: np.ndarray, lstm_delta_standardized: np.ndarray) -> np.ndarray:
    last = sequence_x[:, -1, :]
    first = sequence_x[:, 0, :]
    return np.column_stack([
        lstm_delta_standardized,
        last[:, 0], last[:, 1], last[:, 2], last[:, 3], last[:, 4], last[:, 5], last[:, 6],
        last[:, 7], last[:, 8], last[:, 9], last[:, 10], last[:, 11], last[:, 12], last[:, 13], last[:, 14],
        last[:, 0] - first[:, 0], last[:, 4] - first[:, 4], last[:, 5] - first[:, 5],
    ])


def corrector_inputs(dataset, lstm_delta_standardized: np.ndarray) -> np.ndarray:
    return corrector_inputs_from_sequence(dataset.x, lstm_delta_standardized)


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict:
    errors = predicted - actual
    denominator = np.maximum(np.abs(actual) + np.abs(predicted), 1e-9)
    total = float(np.sum((actual - actual.mean()) ** 2))
    residual = float(np.sum(errors ** 2))
    return {
        "sampleCount": int(actual.size),
        "maeMm": round(float(np.mean(np.abs(errors))), 8),
        "rmseMm": round(float(np.sqrt(np.mean(errors ** 2))), 8),
        "biasMm": round(float(np.mean(errors)), 8),
        "smapePercent": round(float(np.mean(2.0 * np.abs(errors) / denominator) * 100.0), 4),
        "r2": round(1.0 - residual / total, 6) if total > 1e-16 else None,
    }


def ridge_predictions(path: Path, metadata: list[dict]) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {(row["scenario_id"], int(row["origin_hour"]), int(row["target_hour"])): float(row["ridge_prediction_mm"]) for row in rows}
    return np.asarray([mapping[(item["scenario_id"], item["origin_hour"], item["target_hour"])] for item in metadata], dtype=np.float64)


def evaluate(model, dataset, raw_inputs, normalized_inputs, base_y, target_mean, target_std, interval_half_width_mm=None):
    total_y = base_y + model.forward(normalized_inputs)
    raw_prediction = dataset.current + target_mean + total_y * target_std
    prediction = np.maximum(dataset.current, raw_prediction)
    result = {
        "pinn": metric_values(dataset.actual, prediction),
        "pinnUnconstrained": metric_values(dataset.actual, raw_prediction),
        "persistence": metric_values(dataset.actual, dataset.current),
        "constraintCorrections": int(np.sum(raw_prediction < dataset.current)),
        "violationsAfterConstraint": int(np.sum(prediction < dataset.current)),
    }
    if interval_half_width_mm is not None:
        lower = np.maximum(dataset.current, prediction - interval_half_width_mm)
        upper = prediction + interval_half_width_mm
        result["uncertainty"] = {
            "method": "VALIDATION_ABSOLUTE_ERROR_QUANTILE",
            "nominalCoverage": 0.95,
            "halfWidthMm": round(float(interval_half_width_mm), 8),
            "empiricalCoverage": round(float(np.mean((dataset.actual >= lower) & (dataset.actual <= upper))), 4),
            "meanIntervalWidthMm": round(float(np.mean(upper - lower)), 8),
        }
    return result, raw_prediction, prediction


def ablated_prediction(model, dataset, ablated_feature_names, lstm_module, lstm_network, lstm_feature_mean, lstm_feature_std, input_mean, input_std, target_mean, target_std):
    ablated_x = dataset.x.copy()
    for name in ablated_feature_names:
        feature_index = lstm_module.FEATURES.index(name)
        ablated_x[:, :, feature_index] = lstm_feature_mean[0, 0, feature_index]
    ablated_base_y = lstm_network.forward((ablated_x - lstm_feature_mean) / lstm_feature_std)
    ablated_raw_inputs = corrector_inputs_from_sequence(ablated_x, ablated_base_y)
    ablated_normalized_inputs = (ablated_raw_inputs - input_mean) / input_std
    total_y = ablated_base_y + model.forward(ablated_normalized_inputs)
    raw_prediction = dataset.current + target_mean + total_y * target_std
    return np.maximum(dataset.current, raw_prediction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=1, choices=[1, 6])
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=360)
    parser.add_argument("--patience", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0015)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--correction-mode", choices=["monotonic", "free"], default="monotonic",
                        help="Ablación estructural: corrector monótono por construcción o MLP residual sin restricciones monotónicas")
    parser.add_argument("--input-prefix", default="data/generated/ta01-fem-500")
    parser.add_argument("--lstm-model", default=None)
    parser.add_argument("--oof-train-predictions", default=None,
                        help="CSV de predicciones LSTM out-of-fold para ajustar el corrector en entrenamiento")
    parser.add_argument("--ridge-predictions", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--predictions", default=None)
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    lstm_module = load_lstm_module(project_dir)
    prefix = project_dir / args.input_prefix
    datasets = {name: lstm_module.make_sequences(lstm_module.read_csv(Path(f"{prefix}-{name}.csv")), 6, args.horizon) for name in ["train", "validation", "test"]}
    lstm_path = project_dir / (args.lstm_model or f"data/models/ta01-lstm-{args.horizon}h.json")
    ridge_path = project_dir / (args.ridge_predictions or f"data/generated/ta01-baseline-{args.horizon}h-test-predictions.csv")
    lstm_artifact, lstm_network, lstm_feature_mean, lstm_feature_std = load_lstm(lstm_path, lstm_module)
    expected_split = (project_dir / f"{args.input_prefix}-split-manifest.json").resolve()
    lstm_split = (project_dir / lstm_artifact["dataset"]["splitManifest"]).resolve()
    if lstm_split != expected_split:
        raise ValueError("La LSTM y el conjunto del corrector usan particiones distintas")
    target_mean = float(lstm_artifact["model"]["targetDeltaMean"])
    target_std = float(lstm_artifact["model"]["targetDeltaStandardDeviation"])
    base_y = {name: lstm_standardized_delta(lstm_network, dataset, lstm_feature_mean, lstm_feature_std) for name, dataset in datasets.items()}
    oof_path = None
    if args.oof_train_predictions:
        oof_path = (project_dir / args.oof_train_predictions).resolve()
        oof_rows = lstm_module.read_csv(oof_path)
        oof_map = {
            (row["scenario_id"], int(row["origin_hour"]), int(row["target_hour"])): float(row["lstm_delta_standardized"])
            for row in oof_rows
        }
        train_keys = [
            (item["scenario_id"], int(item["origin_hour"]), int(item["target_hour"]))
            for item in datasets["train"].metadata
        ]
        if len(oof_map) != len(oof_rows) or set(oof_map) != set(train_keys):
            raise ValueError("Las predicciones OOF deben cubrir exactamente y sin duplicados las ventanas de entrenamiento")
        base_y["train"] = np.asarray([oof_map[key] for key in train_keys], dtype=np.float64)
    raw_inputs = {name: corrector_inputs(dataset, base_y[name]) for name, dataset in datasets.items()}
    input_mean = raw_inputs["train"].mean(axis=0)
    input_std = raw_inputs["train"].std(axis=0)
    input_std[input_std < 1e-12] = 1.0
    normalized_inputs = {name: (values - input_mean) / input_std for name, values in raw_inputs.items()}
    target_y = {name: (dataset.target_delta - target_mean) / target_std for name, dataset in datasets.items()}

    loss_weights = {
        "data": 1.0,
        "monotonicity": 0.12,
        "rainSensitivity": 0.2 if args.correction_mode == "monotonic" else 0.0,
        "safetySensitivity": 0.2 if args.correction_mode == "monotonic" else 0.0,
        "dryBoundary": 0.08,
        "l2": 1e-6,
    }
    rain_index = CORRECTOR_FEATURES.index("forecast_rainfall_mm")
    safety_index = CORRECTOR_FEATURES.index("mohr_coulomb_safety_index")
    current_rain_index = CORRECTOR_FEATURES.index("current_rainfall_mm_h")
    if args.correction_mode == "monotonic":
        model = PhysicsCorrector(len(CORRECTOR_FEATURES), args.hidden, args.seed, rain_index, safety_index)
    else:
        model = FreeResidualCorrector(len(CORRECTOR_FEATURES), args.hidden, args.seed)
    optimizer = Adam(model.parameters, args.learning_rate)
    random = np.random.default_rng(args.seed)
    best_parameters = copy.deepcopy(model.parameters)
    best_validation_score = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    zero_delta_standardized = target_mean / target_std

    for epoch in range(1, args.epochs + 1):
        order = random.permutation(normalized_inputs["train"].shape[0])
        epoch_losses = []
        for start in range(0, order.size, args.batch_size):
            indices = order[start:start + args.batch_size]
            x = normalized_inputs["train"][indices]
            raw = raw_inputs["train"][indices]
            base = base_y["train"][indices]
            expected = target_y["train"][indices]
            correction, cache = model.forward(x, keep_cache=True)
            total = base + correction
            size = max(1, indices.size)
            gradient = 2.0 * loss_weights["data"] * (total - expected) / size
            loss = loss_weights["data"] * float(np.mean((total - expected) ** 2))

            decreasing = total + zero_delta_standardized < 0
            if np.any(decreasing):
                residual = total[decreasing] + zero_delta_standardized
                gradient[decreasing] += 2.0 * loss_weights["monotonicity"] * residual / size
                loss += loss_weights["monotonicity"] * float(np.sum(residual ** 2) / size)

            dry = (raw[:, rain_index] <= 1e-10) & (raw[:, current_rain_index] <= 1e-10)
            if np.any(dry):
                residual = total[dry] + zero_delta_standardized
                gradient[dry] += 2.0 * loss_weights["dryBoundary"] * residual / size
                loss += loss_weights["dryBoundary"] * float(np.sum(residual ** 2) / size)

            if args.correction_mode == "monotonic":
                rain_raw = raw.copy()
                rain_raw[:, rain_index] += 10.0
                rain_x = (rain_raw - input_mean) / input_std
                rain_correction, rain_cache = model.forward(rain_x, keep_cache=True)
                rain_violation = np.maximum(0.0, correction - rain_correction)
                rain_base_gradient = 2.0 * loss_weights["rainSensitivity"] * rain_violation / size
                rain_perturbed_gradient = -rain_base_gradient
                gradient += rain_base_gradient
                loss += loss_weights["rainSensitivity"] * float(np.mean(rain_violation ** 2))

                safety_raw = raw.copy()
                safety_raw[:, safety_index] += 0.1
                safety_x = (safety_raw - input_mean) / input_std
                safety_correction, safety_cache = model.forward(safety_x, keep_cache=True)
                safety_violation = np.maximum(0.0, safety_correction - correction)
                safety_base_gradient = -2.0 * loss_weights["safetySensitivity"] * safety_violation / size
                safety_perturbed_gradient = -safety_base_gradient
                gradient += safety_base_gradient
                loss += loss_weights["safetySensitivity"] * float(np.mean(safety_violation ** 2))
                gradients = add_gradients(
                    model.backward(gradient, cache),
                    model.backward(rain_perturbed_gradient, rain_cache),
                    model.backward(safety_perturbed_gradient, safety_cache),
                )
            else:
                gradients = model.backward(gradient, cache)
            gradients["w1"] += 2.0 * loss_weights["l2"] * model.parameters["w1"]
            gradients["w2"] += 2.0 * loss_weights["l2"] * model.parameters["w2"]
            optimizer.step(model.parameters, gradients)
            epoch_losses.append(loss)

        validation, _, _ = evaluate(model, datasets["validation"], raw_inputs["validation"], normalized_inputs["validation"], base_y["validation"], target_mean, target_std)
        validation_mae = validation["pinn"]["maeMm"]
        validation_base_correction = model.forward(normalized_inputs["validation"])
        validation_rain_raw = raw_inputs["validation"].copy()
        validation_rain_raw[:, rain_index] += 10.0
        validation_safety_raw = raw_inputs["validation"].copy()
        validation_safety_raw[:, safety_index] += 0.1
        validation_rain_correction = model.forward((validation_rain_raw - input_mean) / input_std)
        validation_safety_correction = model.forward((validation_safety_raw - input_mean) / input_std)
        rain_violation_rate = float(np.mean(validation_rain_correction < validation_base_correction - 1e-12))
        safety_violation_rate = float(np.mean(validation_safety_correction > validation_base_correction + 1e-12))
        validation_score = validation_mae
        if args.correction_mode == "monotonic":
            validation_score += target_std * 0.05 * (rain_violation_rate + safety_violation_rate)
        history.append({
            "epoch": epoch,
            "trainCompositeLoss": round(float(np.mean(epoch_losses)), 8),
            "validationMaeMm": validation_mae,
            "validationRainViolationRate": round(rain_violation_rate, 6),
            "validationSafetyViolationRate": round(safety_violation_rate, 6),
            "selectionScore": round(validation_score, 10),
        })
        if validation_score < best_validation_score - 1e-10:
            best_validation_score = validation_score
            best_epoch = epoch
            best_parameters = copy.deepcopy(model.parameters)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break
    model.parameters = best_parameters

    validation_metrics, _, validation_prediction = evaluate(model, datasets["validation"], raw_inputs["validation"], normalized_inputs["validation"], base_y["validation"], target_mean, target_std)
    calibration_level = 0.95
    calibration_errors = np.sort(np.abs(validation_prediction - datasets["validation"].actual))
    conformal_rank = min(calibration_errors.size - 1, math.ceil((calibration_errors.size + 1) * calibration_level) - 1)
    interval_half_width_mm = float(calibration_errors[conformal_rank])
    evaluations = {}
    predictions = {}
    for name, dataset in datasets.items():
        evaluation, raw_prediction, prediction = evaluate(model, dataset, raw_inputs[name], normalized_inputs[name], base_y[name], target_mean, target_std, interval_half_width_mm)
        evaluation["uncertainty"]["nominalCoverage"] = calibration_level
        evaluation["uncertainty"]["absoluteCalibrationError"] = round(abs(evaluation["uncertainty"]["empiricalCoverage"] - calibration_level), 4)
        _, lstm_prediction = lstm_module.predict_physical(lstm_network, dataset, lstm_feature_mean, lstm_feature_std, target_mean, target_std)
        evaluation["lstm"] = metric_values(dataset.actual, lstm_prediction)
        if name == "test":
            ridge = ridge_predictions(ridge_path, dataset.metadata)
            evaluation["ridgeComparable"] = metric_values(dataset.actual, ridge)
        else:
            ridge = None
        rain_raw = raw_inputs[name].copy()
        rain_raw[:, rain_index] += 10.0
        safety_raw = raw_inputs[name].copy()
        safety_raw[:, safety_index] += 0.1
        base_correction = model.forward(normalized_inputs[name])
        rain_correction = model.forward((rain_raw - input_mean) / input_std)
        safety_correction = model.forward((safety_raw - input_mean) / input_std)
        evaluation["physicsChecks"] = {
            "rainSensitivityViolations": int(np.sum(rain_correction < base_correction - 1e-12)),
            "safetySensitivityViolations": int(np.sum(safety_correction > base_correction + 1e-12)),
            "sampleCount": int(dataset.actual.size),
        }
        evaluations[name] = evaluation
        predictions[name] = (raw_prediction, prediction, lstm_prediction, ridge)

    latency_start = time.perf_counter()
    for _ in range(200):
        model.forward(normalized_inputs["test"][:1])
    latency_ms = (time.perf_counter() - latency_start) * 1000.0 / 200.0

    test_dataset = datasets["test"]
    no_hydrology_features = [
        "current_rainfall_mm_h", "cumulative_rainfall_mm", "forecast_rainfall_mm", "maximum_pore_pressure_kpa",
        "log10_permeability_m_s", "drainage_efficiency", "water_table_m", "storage_coefficient",
    ]
    no_fem_features = [
        "maximum_pore_pressure_kpa", "mohr_coulomb_safety_index", "cohesion_kpa", "friction_angle_deg",
        "unit_weight_kn_m3", "young_modulus_mpa", "log10_permeability_m_s", "drainage_efficiency", "water_table_m", "storage_coefficient",
    ]
    no_hydrology_prediction = ablated_prediction(model, test_dataset, no_hydrology_features, lstm_module, lstm_network, lstm_feature_mean, lstm_feature_std, input_mean, input_std, target_mean, target_std)
    no_fem_prediction = ablated_prediction(model, test_dataset, no_fem_features, lstm_module, lstm_network, lstm_feature_mean, lstm_feature_std, input_mean, input_std, target_mean, target_std)
    ablations = {
        "split": "test",
        "sampleCount": int(test_dataset.actual.size),
        "variants": [
            {"id": "PERSISTENCE", "label": "Persistencia", "components": "estado actual", **evaluations["test"]["persistence"]},
            {"id": "RIDGE", "label": "Ridge", "components": "ventana temporal + FEM + hidrología", **evaluations["test"]["ridgeComparable"]},
            {"id": "LSTM", "label": "LSTM", "components": "LSTM + FEM + hidrología", **evaluations["test"]["lstm"]},
            {"id": "HYBRID_NO_HYDROLOGY", "label": "Híbrido sin hidrología", "components": "LSTM + corrector físico; variables hidrológicas fijadas a la media de entrenamiento", **metric_values(test_dataset.actual, no_hydrology_prediction)},
            {"id": "HYBRID_NO_FEM", "label": "Híbrido sin estado FEM", "components": "LSTM + corrector físico; variables mecánicas fijadas a la media de entrenamiento", **metric_values(test_dataset.actual, no_fem_prediction)},
            {"id": "HYBRID_FULL", "label": "Híbrido físico completo" if args.correction_mode == "monotonic" else "Corrector residual libre", "components": "LSTM + FEM + hidrología + restricciones" if args.correction_mode == "monotonic" else "LSTM + FEM + hidrología; MLP residual sin restricciones monótonas", **evaluations["test"]["pinn"]},
        ],
        "method": "Ablación por sustitución con medias calculadas exclusivamente en entrenamiento",
        "limitation": "Las variables ablacionadas pueden formar combinaciones fuera de la distribución conjunta original; interpretar junto con el análisis de sensibilidad.",
    }
    ablations["bestByMae"] = min(ablations["variants"], key=lambda item: item["maeMm"])["id"]

    output_path = project_dir / (args.output or f"data/models/ta01-physics-guided-{args.horizon}h.json")
    prediction_path = project_dir / (args.predictions or f"data/generated/ta01-physics-guided-{args.horizon}h-test-predictions.csv")
    raw_test, prediction_test, lstm_test, ridge_test = predictions["test"]
    lower = np.maximum(datasets["test"].current, prediction_test - interval_half_width_mm)
    upper = prediction_test + interval_half_width_mm
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["scenario_id", "origin_hour", "target_hour", "actual_displacement_mm", "persistence_prediction_mm", "ridge_prediction_mm", "lstm_prediction_mm", "physics_guided_unconstrained_prediction_mm", "physics_guided_prediction_mm", "interval_lower_mm", "interval_upper_mm", "physics_guided_error_mm"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, item in enumerate(datasets["test"].metadata):
            writer.writerow({
                **item,
                "actual_displacement_mm": f"{datasets['test'].actual[index]:.8f}",
                "persistence_prediction_mm": f"{datasets['test'].current[index]:.8f}",
                "ridge_prediction_mm": f"{ridge_test[index]:.8f}",
                "lstm_prediction_mm": f"{lstm_test[index]:.8f}",
                "physics_guided_unconstrained_prediction_mm": f"{raw_test[index]:.8f}",
                "physics_guided_prediction_mm": f"{prediction_test[index]:.8f}",
                "interval_lower_mm": f"{lower[index]:.8f}",
                "interval_upper_mm": f"{upper[index]:.8f}",
                "physics_guided_error_mm": f"{prediction_test[index] - datasets['test'].actual[index]:.8f}",
            })

    result = {
        "id": f"TA01-PHYSICS-GUIDED-{args.horizon}H-SEED{args.seed}",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "method": "LSTM_RESIDUAL_PHYSICS_GUIDED_MLP_V1" if args.correction_mode == "monotonic" else "LSTM_RESIDUAL_FREE_MLP_ABLATION_V1",
        "scientificStatus": "RED_NEURONAL_GUIADA_POR_FISICA_AGREGADA_NO_PINN_PDE_NO_OPERACIONAL",
        "scope": "Corrector temporal agregado sobre LSTM. No resuelve el equilibrio espacial ni sustituye una PINN de campo.",
        "dataset": lstm_artifact["dataset"],
        "architecture": {
            "type": "LSTM + MLP residual monótono guiado por física" if args.correction_mode == "monotonic" else "LSTM + MLP residual sin restricciones monótonas",
            "horizonHours": args.horizon,
            "lookbackHours": 6,
            "inputFeatures": CORRECTOR_FEATURES,
            "hiddenUnits": args.hidden,
            "baseModelId": lstm_artifact["id"],
            "output": "corrección del incremento estandarizado de desplazamiento inducido por lluvia",
            "correctionMode": args.correction_mode,
            "monotonicByConstruction": {
                "forecast_rainfall_mm": "NON_DECREASING",
                "mohr_coulomb_safety_index": "NON_INCREASING",
            } if args.correction_mode == "monotonic" else {},
        },
        "training": {
            "optimizer": "Adam",
            "epochsRequested": args.epochs,
            "epochsCompleted": len(history),
            "bestEpoch": best_epoch,
            "earlyStoppingPatience": args.patience,
            "batchSize": args.batch_size,
            "learningRate": args.learning_rate,
            "seed": args.seed,
            "baseTrainingPredictions": "OUT_OF_FOLD" if oof_path else "IN_SAMPLE",
            "oofPredictionFile": str(oof_path.relative_to(project_dir)) if oof_path else None,
            "lossWeights": loss_weights,
            "lossTerms": ["FEM supervised consistency", "monotonic accumulated displacement", "rainfall sensitivity", "safety-index sensitivity", "dry-boundary residual", "L2"] if args.correction_mode == "monotonic" else ["FEM supervised consistency", "monotonic accumulated displacement", "dry-boundary residual", "L2"],
            "selectionMetric": "MAE de validación más penalización por violaciones condicionales de lluvia y seguridad" if args.correction_mode == "monotonic" else "MAE de validación",
        },
        "uncertainty": {
            "method": "VALIDATION_ABSOLUTE_ERROR_QUANTILE",
            "nominalCoverage": calibration_level,
            "halfWidthMm": interval_half_width_mm,
        },
        "latency": {"meanSingleInferenceMs": round(latency_ms, 6), "benchmarkRuns": 200, "environment": "NumPy CPU"},
        "metrics": evaluations,
        "ablation": ablations,
        "history": history,
        "model": {
            "inputMean": dict(zip(CORRECTOR_FEATURES, serializable_array(input_mean))),
            "inputStandardDeviation": dict(zip(CORRECTOR_FEATURES, serializable_array(input_std))),
            "weights": {name: serializable_array(value) for name, value in model.parameters.items()},
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model": str(output_path.relative_to(project_dir)),
        "predictions": str(prediction_path.relative_to(project_dir)),
        "bestEpoch": best_epoch,
        "epochsCompleted": len(history),
        "latencyMs": round(latency_ms, 6),
        "test": evaluations["test"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
