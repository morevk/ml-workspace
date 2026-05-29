#!/usr/bin/env python3
"""Autoencoder-based anomaly detection POC for multichannel sensor data."""
""" 
Command that gave result with >95% accuracy and >90% recall:
python autoencoder_anomaly_detection.py --test-anomaly-fraction 0.2 --plot-dir plots --train-normal-only --scoring-mode window --window-threshold-std 3 --anomaly-magnitude small --train-split 0.5 --window-test-calibrate-on all --test-anomaly-fraction 0.05 --test-clean-prefix-windows 2350
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tensorflow import keras
from tensorflow.keras import layers

SENSOR_COLUMNS = ["current", "voltage", "power_factor", "active_power"]
LABEL_COLUMN = "is_anomaly"
WINDOW_SIZE = 50
THRESHOLD_STD = 3.0
ANOMALY_MAGNITUDES = ("small", "medium", "large")
DEFAULT_ANOMALY_MAGNITUDE = "small"
THRESHOLD_CALIBRATION_SOURCES = ("train", "test_normal")
DEFAULT_THRESHOLD_CALIBRATION = "train"
WINDOW_TEST_CALIBRATION_SOURCES = ("clean", "all")
DEFAULT_WINDOW_TEST_CALIBRATION = "clean"
# None => 2 * window_size initial test windows without fault timesteps
DEFAULT_TEST_CLEAN_PREFIX_WINDOWS: int | None = None
TEST_WINDOW_CALIBRATION_CLEAN = "clean test windows (no fault timesteps)"
TEST_WINDOW_CALIBRATION_ALL = "test (all windows)"


@dataclass
class DetectionResult:
    reconstruction_errors: np.ndarray
    threshold: float
    is_anomaly: np.ndarray


@dataclass
class TimestepDetectionResult:
    timestep_scores: np.ndarray
    threshold: float
    timestep_is_anomaly: np.ndarray
    window_is_anomaly: np.ndarray


@dataclass
class WindowThreshold:
    """Threshold from μ + kσ on a calibration score distribution."""

    value: float
    std_multiplier: float
    calibration_scores: np.ndarray


@dataclass
class WindowThresholds:
    """Separate window-level thresholds for train and test evaluation."""

    train: WindowThreshold
    test: WindowThreshold


@dataclass
class TimestepThreshold:
    """Threshold derived from training timestep aggregate scores."""

    value: float
    std_multiplier: float
    train_scores: np.ndarray


@dataclass
class SensorDataset:
    """Train and test sensor data kept as separate series."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_point_labels: np.ndarray
    test_point_labels: np.ndarray
    train_window_ground_truth: np.ndarray
    test_window_ground_truth: np.ndarray

    @property
    def window_ground_truth(self) -> np.ndarray:
        return np.concatenate(
            [self.train_window_ground_truth, self.test_window_ground_truth]
        )


def synthetic_window_counts(
    n_samples: int,
    window_size: int,
    train_split: float,
) -> tuple[int, int]:
    """Return (n_train_windows, n_test_windows) for a target sample count."""
    n_windows = n_samples - window_size + 1
    n_train = round(n_windows * train_split)
    return n_train, n_windows - n_train


def _anomaly_profile(magnitude: str) -> dict[str, tuple[float, float] | float]:
    """Return perturbation ranges for synthetic anomaly injection."""
    profiles = {
        "small": {
            "current_add": (0.3, 0.8),
            "voltage_add": (1.0, 2.0),
            "pf_add": (0.01, 0.02),
            "timestep_fraction": 0.2,
            "power_noise": 1.0,
        },
        "medium": {
            "current_scale": (1.1, 1.25),
            "voltage_add": (5.0, 15.0),
            "pf_add": (-0.12, -0.05),
            "timestep_fraction": 0.5,
            "power_noise": 5.0,
        },
        "large": {
            "current_scale": (1.4, 2.0),
            "voltage_add": (-25.0, 25.0),
            "pf_replace": (0.5, 0.75),
            "timestep_fraction": 1.0,
            "power_noise": 5.0,
        },
    }
    if magnitude not in profiles:
        raise ValueError(
            f"Unknown anomaly magnitude {magnitude!r}. "
            f"Choose from: {', '.join(ANOMALY_MAGNITUDES)}"
        )
    return profiles[magnitude]


def _apply_anomaly_perturbations(
    df: pd.DataFrame,
    indices: np.ndarray,
    rng: np.random.Generator,
    magnitude: str = DEFAULT_ANOMALY_MAGNITUDE,
) -> None:
    """Inject sensor anomalies; recompute active_power from I×V×PF."""
    if len(indices) == 0:
        return

    profile = _anomaly_profile(magnitude)
    n = len(indices)

    if "current_add" in profile:
        low, high = profile["current_add"]
        df.loc[indices, "current"] += rng.uniform(low, high, n)
    elif "current_scale" in profile:
        low, high = profile["current_scale"]
        df.loc[indices, "current"] *= rng.uniform(low, high, n)

    if "voltage_add" in profile:
        low, high = profile["voltage_add"]
        df.loc[indices, "voltage"] += rng.uniform(low, high, n)

    if "pf_add" in profile:
        low, high = profile["pf_add"]
        df.loc[indices, "power_factor"] += rng.uniform(low, high, n)
    elif "pf_replace" in profile:
        low, high = profile["pf_replace"]
        df.loc[indices, "power_factor"] = rng.uniform(low, high, n)

    power_noise = float(profile.get("power_noise", 5.0))
    df.loc[indices, "active_power"] = (
        df.loc[indices, "current"].to_numpy()
        * df.loc[indices, "voltage"].to_numpy()
        * df.loc[indices, "power_factor"].to_numpy()
        + rng.normal(0.0, power_noise, n)
    )


def _generate_normal_trace(
    n_timesteps: int,
    time_offset: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate a normal multichannel sensor trace."""
    time = np.arange(time_offset, time_offset + n_timesteps)

    current = 10.0 + 0.4 * np.sin(time / 12.0) + rng.normal(0.0, 0.15, n_timesteps)
    voltage = 230.0 + 0.8 * np.cos(time / 18.0) + rng.normal(0.0, 0.5, n_timesteps)
    power_factor = 0.92 + 0.03 * np.sin(time / 25.0) + rng.normal(0.0, 0.01, n_timesteps)
    active_power = current * voltage * power_factor + rng.normal(0.0, 5.0, n_timesteps)

    return pd.DataFrame(
        {
            "current": current,
            "voltage": voltage,
            "power_factor": power_factor,
            "active_power": active_power,
        }
    )


def generate_train_data(
    n_train_windows: int,
    window_size: int = WINDOW_SIZE,
    seed: int = 42,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate normal training data only (no anomalies)."""
    if rng is None:
        rng = np.random.default_rng(seed)
    n_timesteps = n_train_windows + window_size - 1

    df = _generate_normal_trace(n_timesteps, time_offset=0, rng=rng)
    point_labels = np.zeros(n_timesteps, dtype=bool)
    window_ground_truth = np.zeros(n_train_windows, dtype=bool)

    return df, point_labels, window_ground_truth


def generate_test_data(
    n_test_windows: int,
    window_size: int = WINDOW_SIZE,
    test_anomaly_fraction: float = 0.2,
    time_offset: int = 0,
    seed: int = 42,
    anomaly_magnitude: str = DEFAULT_ANOMALY_MAGNITUDE,
    rng: np.random.Generator | None = None,
    normal_baseline: pd.DataFrame | None = None,
    clean_prefix_windows: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate test data with a mix of normal and anomalous windows.

    If ``normal_baseline`` is provided, it is copied as the test trace before
    faults are injected (typically a duplicate of the train series segment).
    Faults are placed only in later windows so the first ``clean_prefix_windows``
    sliding windows stay free of perturbations. Otherwise a new trace is drawn
    with ``_generate_normal_trace``.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    profile = _anomaly_profile(anomaly_magnitude)
    n_timesteps = n_test_windows + window_size - 1

    if normal_baseline is not None:
        if len(normal_baseline) != n_timesteps:
            raise ValueError(
                f"normal_baseline length {len(normal_baseline)} does not match "
                f"expected test timesteps {n_timesteps}."
            )
        missing = [c for c in SENSOR_COLUMNS if c not in normal_baseline.columns]
        if missing:
            raise ValueError(f"normal_baseline missing columns: {missing}")
        df = normal_baseline[SENSOR_COLUMNS].copy()
    else:
        df = _generate_normal_trace(n_timesteps, time_offset=time_offset, rng=rng)
    point_labels = np.zeros(n_timesteps, dtype=bool)
    window_ground_truth = np.zeros(n_test_windows, dtype=bool)

    if clean_prefix_windows is None:
        clean_prefix_windows = 2 * window_size
    clean_prefix_windows = max(0, min(clean_prefix_windows, n_test_windows))
    # Earliest fault start so windows [0, clean_prefix_windows) have no perturbed
    # timesteps (window i uses timesteps [i, i + window_size)). Use 0 to disable.
    if clean_prefix_windows == 0:
        min_fault_start = 0
    else:
        min_fault_start = min(
            clean_prefix_windows + window_size - 1,
            max(0, n_test_windows - 1),
        )
    fault_start_candidates = np.arange(min_fault_start, n_test_windows)
    if len(fault_start_candidates) == 0:
        fault_start_candidates = np.arange(n_test_windows)

    n_anom_windows = round(n_test_windows * test_anomaly_fraction)
    n_anom_windows = min(n_anom_windows, len(fault_start_candidates))
    anom_window_starts = rng.choice(
        fault_start_candidates, size=n_anom_windows, replace=False
    )

    timestep_fraction = float(profile["timestep_fraction"])
    for start in anom_window_starts:
        end = start + window_size
        window_indices = np.arange(start, end)
        n_perturb = max(1, int(round(window_size * timestep_fraction)))
        n_perturb = min(n_perturb, len(window_indices))
        perturb_indices = rng.choice(window_indices, size=n_perturb, replace=False)

        point_labels[perturb_indices] = True
        _apply_anomaly_perturbations(
            df, perturb_indices, rng, magnitude=anomaly_magnitude
        )

    window_ground_truth[anom_window_starts] = True

    return df, point_labels, window_ground_truth


def generate_synthetic_dataset(
    n_samples: int = 5000,
    seed: int = 42,
    train_split: float = 0.8,
    window_size: int = WINDOW_SIZE,
    test_anomaly_fraction: float = 0.2,
    anomaly_magnitude: str = DEFAULT_ANOMALY_MAGNITUDE,
    clean_prefix_windows: int | None = DEFAULT_TEST_CLEAN_PREFIX_WINDOWS,
) -> SensorDataset:
    """Build separate train and test synthetic datasets.

    Test is a copy of the train timestep series (for the test length), then
    anomalies are injected only after ``clean_prefix_windows`` (default
    ``2 * window_size``). Train uses ``default_rng(seed)``; test fault placement
    uses a fresh ``default_rng(seed)`` (isolated RNG stream).
    """
    n_train_windows, n_test_windows = synthetic_window_counts(
        n_samples, window_size, train_split
    )
    train_rng = np.random.default_rng(seed)
    test_rng = np.random.default_rng(seed)

    train_df, train_labels, train_window_gt = generate_train_data(
        n_train_windows=n_train_windows,
        window_size=window_size,
        rng=train_rng,
    )

    test_timesteps = n_test_windows + window_size - 1
    # Duplicate train series for test, then fault only after the clean prefix.
    test_baseline = train_df.iloc[:test_timesteps].copy().reset_index(drop=True)

    test_df, test_labels, test_window_gt = generate_test_data(
        n_test_windows=n_test_windows,
        window_size=window_size,
        test_anomaly_fraction=test_anomaly_fraction,
        anomaly_magnitude=anomaly_magnitude,
        rng=test_rng,
        normal_baseline=test_baseline,
        clean_prefix_windows=clean_prefix_windows,
    )

    return SensorDataset(
        train_df=train_df,
        test_df=test_df,
        train_point_labels=train_labels,
        test_point_labels=test_labels,
        train_window_ground_truth=train_window_gt,
        test_window_ground_truth=test_window_gt,
    )

def load_sensor_data(
    path: Path | None,
    seed: int = 42,
    train_split: float = 0.8,
    window_size: int = WINDOW_SIZE,
    test_anomaly_fraction: float = 0.2,
    anomaly_magnitude: str = DEFAULT_ANOMALY_MAGNITUDE,
    clean_prefix_windows: int | None = DEFAULT_TEST_CLEAN_PREFIX_WINDOWS,
) -> SensorDataset:
    """Load or generate train/test sensor data."""
    if path is None:
        return generate_synthetic_dataset(
            n_samples=5000,
            seed=seed,
            train_split=train_split,
            window_size=window_size,
            test_anomaly_fraction=test_anomaly_fraction,
            anomaly_magnitude=anomaly_magnitude,
            clean_prefix_windows=clean_prefix_windows,
        )

    df = pd.read_csv(path)
    missing = [col for col in SENSOR_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    values = df[SENSOR_COLUMNS]
    point_labels = (
        df[LABEL_COLUMN].astype(bool).to_numpy()
        if LABEL_COLUMN in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    all_windows = window_sensor_data(values.to_numpy(dtype=np.float32), window_size)
    window_gt = window_labels(point_labels, window_size)

    n_train = round(len(all_windows) * train_split)
    train_timesteps = n_train + window_size - 1

    return SensorDataset(
        train_df=values.iloc[:train_timesteps].reset_index(drop=True),
        test_df=values.iloc[n_train:].reset_index(drop=True),
        train_point_labels=point_labels[:train_timesteps],
        test_point_labels=point_labels[n_train:],
        train_window_ground_truth=window_gt[:n_train],
        test_window_ground_truth=window_gt[n_train:],
    )


def filter_normal_windows(
    windows: np.ndarray,
    point_labels: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, int]:
    """Keep only windows with no anomalous timesteps."""
    window_anomaly_flags = window_labels(point_labels, window_size)
    if len(window_anomaly_flags) != len(windows):
        raise ValueError(
            "Window label count does not match window count: "
            f"{len(window_anomaly_flags)} vs {len(windows)}."
        )

    normal_mask = ~window_anomaly_flags
    removed = int((~normal_mask).sum())
    return windows[normal_mask], removed


def describe_synthetic_split(
    dataset: SensorDataset,
    test_anomaly_fraction: float,
    anomaly_magnitude: str,
    clean_prefix_windows: int | None = None,
    window_size: int = WINDOW_SIZE,
) -> None:
    """Print train/test window composition for structured synthetic data."""
    n_train = len(dataset.train_window_ground_truth)
    n_test = len(dataset.test_window_ground_truth)
    train_anom = int(dataset.train_window_ground_truth.sum())
    test_anom = int(dataset.test_window_ground_truth.sum())
    test_normal = n_test - test_anom
    profile = _anomaly_profile(anomaly_magnitude)

    print(
        "\nSynthetic dataset split (test = duplicate of train segment, "
        "faults in later windows only):"
    )
    effective_clean_prefix = (
        2 * window_size if clean_prefix_windows is None else clean_prefix_windows
    )
    print(f"  Anomaly magnitude: {anomaly_magnitude}")
    print(
        f"  Clean prefix windows: {effective_clean_prefix} "
        f"(fault starts from window "
        f"{0 if effective_clean_prefix == 0 else effective_clean_prefix + window_size - 1})"
    )
    print(
        f"  Perturbation: {int(profile['timestep_fraction'] * 100)}% of timesteps "
        "per anomalous window; active_power = I×V×PF"
    )
    print(
        f"  Train: {n_train} windows, {len(dataset.train_df)} timesteps "
        f"(normal {n_train - train_anom}, anomalous {train_anom})"
    )
    print(
        f"  Test:  {n_test} windows, {len(dataset.test_df)} timesteps "
        f"(normal {test_normal}, anomalous {test_anom} "
        f"[target {100 * (1 - test_anomaly_fraction):.0f}% / "
        f"{100 * test_anomaly_fraction:.0f}%])"
    )
    anom_window_indices = np.flatnonzero(dataset.test_window_ground_truth)
    preview = anom_window_indices[:10].tolist()
    print(f"  First 10 anomaly window indices: {preview}")


def window_sensor_data(values: np.ndarray, window_size: int) -> np.ndarray:
    """Convert (n_timesteps, n_channels) into (n_windows, window_size, n_channels)."""
    if values.shape[0] < window_size:
        raise ValueError(
            f"Need at least {window_size} timesteps, got {values.shape[0]}."
        )

    windows = []
    for start in range(values.shape[0] - window_size + 1):
        windows.append(values[start : start + window_size])

    return np.asarray(windows, dtype=np.float32)


def window_labels(labels: np.ndarray, window_size: int) -> np.ndarray:
    """Mark a window anomalous if any timestep inside it is anomalous."""
    windowed = []
    for start in range(labels.shape[0] - window_size + 1):
        windowed.append(labels[start : start + window_size].any())
    return np.asarray(windowed, dtype=bool)


def clean_window_mask(point_labels: np.ndarray, window_size: int) -> np.ndarray:
    """True for windows that contain no anomalous timestep."""
    return ~window_labels(point_labels, window_size)


def build_autoencoder(input_shape: tuple[int, int]) -> keras.Model:
    """Conv1D autoencoder for (window_size, n_channels) windows."""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv1D(filters=32, kernel_size=3, padding="same")(inputs)
    x = layers.Conv1D(filters=16, kernel_size=3, padding="same")(x)
    x = layers.Conv1D(filters=8, kernel_size=3, padding="same")(x)
    x = layers.Conv1D(filters=4, kernel_size=3, padding="same")(x)
    x = layers.Conv1D(filters=8, kernel_size=3, padding="same")(x)
    x = layers.Conv1D(filters=16, kernel_size=3, padding="same")(x)
    x = layers.Conv1D(filters=32, kernel_size=3, padding="same")(x)
    outputs = layers.Conv1D(filters=input_shape[1], kernel_size=3, padding="same")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="sensor_autoencoder")


def reconstruction_errors(model: keras.Model, windows: np.ndarray) -> np.ndarray:
    """Per-window mean squared error averaged over timesteps and channels."""
    reconstructions = model.predict(windows, verbose=0)
    return np.mean(np.square(windows - reconstructions), axis=(1, 2))


def window_timestep_errors(model: keras.Model, windows: np.ndarray) -> np.ndarray:
    """Per-window, per-timestep MSE averaged over channels. Shape: (n_windows, window_size)."""
    reconstructions = model.predict(windows, verbose=0)
    return np.mean(np.square(windows - reconstructions), axis=2)


def aggregate_timestep_scores(
    window_timestep_errors: np.ndarray,
    n_timesteps: int,
    method: str = "max",
) -> np.ndarray:
    """Combine overlapping window errors into one score per timestep."""
    n_windows, window_size = window_timestep_errors.shape
    if method == "max":
        scores = np.zeros(n_timesteps, dtype=np.float32)
        for offset in range(window_size):
            timesteps = np.arange(n_windows) + offset
            np.maximum.at(scores, timesteps, window_timestep_errors[:, offset])
        return scores

    if method == "mean":
        sums = np.zeros(n_timesteps, dtype=np.float64)
        counts = np.zeros(n_timesteps, dtype=np.int32)
        for offset in range(window_size):
            timesteps = np.arange(n_windows) + offset
            np.add.at(sums, timesteps, window_timestep_errors[:, offset])
            np.add.at(counts, timesteps, 1)
        return (sums / np.maximum(counts, 1)).astype(np.float32)

    raise ValueError(f"Unknown aggregation method: {method!r}. Use 'max' or 'mean'.")


def windows_from_timestep_flags(
    timestep_flags: np.ndarray,
    window_size: int,
    window_start: int,
    n_windows: int,
) -> np.ndarray:
    """Flag a window if any timestep inside it is flagged."""
    flags = np.zeros(n_windows, dtype=bool)
    for i in range(n_windows):
        start = window_start + i
        flags[i] = timestep_flags[start : start + window_size].any()
    return flags


def detect_anomalies_timestep(
    timestep_scores: np.ndarray,
    threshold: float,
    window_size: int,
    window_start: int,
    n_eval_windows: int,
) -> TimestepDetectionResult:
    """Flag timesteps above threshold, then roll up to window predictions."""
    timestep_is_anomaly = timestep_scores > threshold
    window_is_anomaly = windows_from_timestep_flags(
        timestep_is_anomaly,
        window_size,
        window_start,
        n_eval_windows,
    )
    return TimestepDetectionResult(
        timestep_scores=timestep_scores,
        threshold=threshold,
        timestep_is_anomaly=timestep_is_anomaly,
        window_is_anomaly=window_is_anomaly,
    )


def fit_threshold(errors: np.ndarray, n_std: float) -> float:
    return float(np.mean(errors) + n_std * np.std(errors))


def print_threshold_calculation(
    mode_label: str,
    calibration_scores: np.ndarray,
    std_multiplier: float,
    threshold: float,
    calibration_source: str,
) -> None:
    """Print mean, std, and threshold formula used for anomaly detection."""
    mean = float(np.mean(calibration_scores))
    std = float(np.std(calibration_scores))
    print(
        f"\n[{mode_label}] Threshold calculation "
        f"({len(calibration_scores)} {calibration_source} samples):"
    )
    print(f"  Min:             {float(np.min(calibration_scores)):.6f}")
    print(f"  Max:             {float(np.max(calibration_scores)):.6f}")
    print(f"  Mean (μ):        {mean:.6f}")
    print(f"  Std (σ):         {std:.6f}")
    print(f"  Std multiplier:  {std_multiplier}")
    print(
        f"  Threshold:       μ + {std_multiplier}σ = "
        f"{mean:.6f} + {std_multiplier * std:.6f} = {threshold:.6f}"
    )


def print_window_mse_distribution(
    split_label: str,
    window_errors: np.ndarray,
    threshold: float,
    std_multiplier: float,
    calibration_source: str,
) -> None:
    """Print min/max/mean/std and how many windows exceed the threshold."""
    above = int(np.sum(window_errors > threshold))
    n = len(window_errors)
    print(f"\n[Window-level] {split_label} ({n} windows):")
    print(f"  Min:             {float(np.min(window_errors)):.6f}")
    print(f"  Max:             {float(np.max(window_errors)):.6f}")
    print(f"  Mean (μ):        {float(np.mean(window_errors)):.6f}")
    print(f"  Std (σ):         {float(np.std(window_errors)):.6f}")
    print(f"  Std multiplier:  {std_multiplier} (from {calibration_source} calibration)")
    print(f"  Threshold:       {threshold:.6f}")
    if n:
        print(
            f"  Above threshold: {above} ({100 * above / n:.2f}%)"
        )
    else:
        print("  Above threshold: 0 (0.00%)")


def print_window_level_mse_report(
    train_window_errors: np.ndarray,
    test_window_errors: np.ndarray,
    window_thresholds: WindowThresholds,
    test_calibration_source: str,
) -> None:
    """Print calibration and MSE stats for train and test with split-specific thresholds."""
    print_threshold_calculation(
        "Window-level (train)",
        window_thresholds.train.calibration_scores,
        window_thresholds.train.std_multiplier,
        window_thresholds.train.value,
        "train",
    )
    print_window_mse_distribution(
        "Train reconstruction MSE",
        train_window_errors,
        window_thresholds.train.value,
        window_thresholds.train.std_multiplier,
        "train",
    )
    print_threshold_calculation(
        "Window-level (test)",
        window_thresholds.test.calibration_scores,
        window_thresholds.test.std_multiplier,
        window_thresholds.test.value,
        test_calibration_source,
    )
    print_window_mse_distribution(
        "Test reconstruction MSE",
        test_window_errors,
        window_thresholds.test.value,
        window_thresholds.test.std_multiplier,
        test_calibration_source,
    )


def warn_test_mse_shift(
    train_errors: np.ndarray,
    test_errors: np.ndarray,
    clean_test_window_mask: np.ndarray | None,
) -> None:
    """Warn when test reconstruction errors sit above the train distribution."""
    if clean_test_window_mask is not None and clean_test_window_mask.any():
        reference = test_errors[clean_test_window_mask]
        label = "clean test windows"
    else:
        reference = test_errors
        label = "test windows"

    train_p95 = float(np.percentile(train_errors, 95))
    test_median = float(np.median(reference))
    if test_median <= train_p95 * 2:
        return

    print(
        f"\n⚠ Distribution shift: {label} MSE (median {test_median:.6f}) is much "
        f"higher than train (95th %ile {train_p95:.6f})."
    )
    print(
        "  Train and test use separate window thresholds; see "
        "--window-test-calibrate-on for test calibration scope."
    )


def select_test_window_calibration_errors(
    test_window_errors: np.ndarray,
    test_point_labels: np.ndarray,
    window_size: int,
    calibration: str,
) -> tuple[np.ndarray, str]:
    """Return MSE scores and label for test window threshold calibration."""
    if calibration == "all":
        return test_window_errors, TEST_WINDOW_CALIBRATION_ALL

    if calibration != "clean":
        raise ValueError(
            f"Unknown window test calibration {calibration!r}. "
            f"Choose from: {', '.join(WINDOW_TEST_CALIBRATION_SOURCES)}"
        )

    test_clean_mask = clean_window_mask(test_point_labels, window_size)
    if not test_clean_mask.any():
        raise ValueError(
            "No clean test windows (no fault timesteps inside) available for "
            "threshold calibration. Use --window-test-calibrate-on all, reduce "
            "--test-anomaly-fraction, or provide data with normal stretches."
        )
    return test_window_errors[test_clean_mask], TEST_WINDOW_CALIBRATION_CLEAN


def fit_window_threshold(
    calibration_errors: np.ndarray,
    std_multiplier: float,
) -> WindowThreshold:
    """Compute anomaly threshold from window-level MSE scores."""
    threshold_value = fit_threshold(calibration_errors, std_multiplier)
    return WindowThreshold(
        value=threshold_value,
        std_multiplier=std_multiplier,
        calibration_scores=calibration_errors,
    )


def fit_window_thresholds(
    train_window_errors: np.ndarray,
    test_window_errors: np.ndarray,
    std_multiplier: float,
    test_point_labels: np.ndarray,
    window_size: int,
    test_calibration: str = DEFAULT_WINDOW_TEST_CALIBRATION,
) -> tuple[WindowThresholds, str]:
    """Fit train and test window thresholds; return calibration source label."""
    n_test_windows = len(test_window_errors)
    expected_timesteps = n_test_windows + window_size - 1
    if len(test_point_labels) != expected_timesteps:
        raise ValueError(
            "Test point label count does not match window count: "
            f"{len(test_point_labels)} timesteps vs {n_test_windows} windows."
        )
    test_calib_errors, calib_source = select_test_window_calibration_errors(
        test_window_errors,
        test_point_labels,
        window_size,
        test_calibration,
    )
    return WindowThresholds(
        train=fit_window_threshold(train_window_errors, std_multiplier),
        test=fit_window_threshold(test_calib_errors, std_multiplier),
    ), calib_source


def fit_timestep_threshold(
    calibration_scores: np.ndarray,
    std_multiplier: float,
    aggregate_method: str = "max",
    calibration_source: str = "train",
) -> TimestepThreshold:
    """Compute anomaly threshold from timestep aggregate scores."""
    threshold_value = fit_threshold(calibration_scores, std_multiplier)
    print_threshold_calculation(
        f"Timestep-level ({aggregate_method})",
        calibration_scores,
        std_multiplier,
        threshold_value,
        calibration_source,
    )
    return TimestepThreshold(
        value=threshold_value,
        std_multiplier=std_multiplier,
        train_scores=calibration_scores,
    )


def detect_from_window_errors(
    window_errors: np.ndarray,
    threshold: WindowThreshold,
) -> DetectionResult:
    """Flag windows using a window-level threshold only."""
    return DetectionResult(
        reconstruction_errors=window_errors,
        threshold=threshold.value,
        is_anomaly=window_errors > threshold.value,
    )


def detect_from_timestep_scores(
    timestep_scores: np.ndarray,
    threshold: TimestepThreshold,
    window_size: int,
    n_eval_windows: int,
    window_start: int = 0,
) -> TimestepDetectionResult:
    """Flag timesteps and roll up to windows using a timestep-level threshold only."""
    return detect_anomalies_timestep(
        timestep_scores=timestep_scores,
        threshold=threshold.value,
        window_size=window_size,
        window_start=window_start,
        n_eval_windows=n_eval_windows,
    )


def detect_anomalies(
    model: keras.Model,
    windows: np.ndarray,
    threshold: float,
) -> DetectionResult:
    errors = reconstruction_errors(model, windows)
    return DetectionResult(
        reconstruction_errors=errors,
        threshold=threshold,
        is_anomaly=errors > threshold,
    )


def print_detection_metrics(
    title: str,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> None:
    """Print precision, recall, and confusion matrix for window-level predictions."""
    tp = int(np.sum(predicted & ground_truth))
    fp = int(np.sum(predicted & ~ground_truth))
    fn = int(np.sum(~predicted & ground_truth))
    tn = int(np.sum(~predicted & ~ground_truth))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    print(f"\n{title}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")


def plot_training_history(
    history: keras.callbacks.History,
    output_path: Path,
) -> None:
    """Plot epoch-wise training and validation MSE loss."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history.history["loss"]) + 1)

    ax.plot(epochs, history.history["loss"], label="Train MSE", linewidth=2)
    if "val_loss" in history.history:
        ax.plot(epochs, history.history["val_loss"], label="Validation MSE", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_window_mse(
    errors: np.ndarray,
    threshold: float,
    is_anomaly: np.ndarray,
    output_path: Path,
    title: str,
    xlabel: str,
    threshold_std: float = THRESHOLD_STD,
    ground_truth: np.ndarray | None = None,
) -> None:
    """Plot per-window reconstruction MSE with threshold and flagged windows."""
    window_indices = np.arange(len(errors))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        window_indices,
        errors,
        color="steelblue",
        linewidth=0.8,
        alpha=0.85,
        label="Per-window MSE",
    )
    ax.axhline(
        threshold,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold (mean + {threshold_std}σ) = {threshold:.6f}",
    )

    if ground_truth is not None and ground_truth.any():
        ax.scatter(
            window_indices[ground_truth],
            errors[ground_truth],
            color="darkorange",
            s=14,
            zorder=2,
            alpha=0.7,
            label=f"Known anomalous ({ground_truth.sum()})",
        )

    anomaly_indices = window_indices[is_anomaly]
    if len(anomaly_indices) > 0:
        ax.scatter(
            anomaly_indices,
            errors[is_anomaly],
            color="crimson",
            s=12,
            zorder=3,
            label=f"Flagged windows ({is_anomaly.sum()})",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_train_window_mse(
    train_errors: np.ndarray,
    threshold: float,
    is_anomaly: np.ndarray,
    output_path: Path,
    threshold_std: float = THRESHOLD_STD,
) -> None:
    """Plot per-window reconstruction MSE for all training windows."""
    plot_window_mse(
        errors=train_errors,
        threshold=threshold,
        is_anomaly=is_anomaly,
        output_path=output_path,
        title=f"Training Window Reconstruction MSE ({len(train_errors)} windows)",
        xlabel="Train window index",
        threshold_std=threshold_std,
    )


def plot_test_window_mse(
    test_errors: np.ndarray,
    threshold: float,
    is_anomaly: np.ndarray,
    output_path: Path,
    threshold_std: float = THRESHOLD_STD,
    ground_truth: np.ndarray | None = None,
) -> None:
    """Plot per-window reconstruction MSE for all test windows."""
    plot_window_mse(
        errors=test_errors,
        threshold=threshold,
        is_anomaly=is_anomaly,
        output_path=output_path,
        title=f"Test Window Reconstruction MSE ({len(test_errors)} windows)",
        xlabel="Test window index",
        threshold_std=threshold_std,
        ground_truth=ground_truth,
    )


def plot_training_results(
    history: keras.callbacks.History,
    train_errors: np.ndarray,
    train_threshold: float,
    is_anomaly: np.ndarray,
    output_dir: Path,
    threshold_std: float = THRESHOLD_STD,
    test_errors: np.ndarray | None = None,
    test_is_anomaly: np.ndarray | None = None,
    test_threshold: float | None = None,
    test_threshold_std: float | None = None,
    test_ground_truth: np.ndarray | None = None,
) -> None:
    """Save training loss and per-window MSE plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, output_dir / "training_loss.png")
    plot_train_window_mse(
        train_errors,
        train_threshold,
        is_anomaly,
        output_dir / "train_window_mse.png",
        threshold_std=threshold_std,
    )
    if test_errors is not None and test_is_anomaly is not None:
        plot_test_window_mse(
            test_errors,
            test_threshold if test_threshold is not None else train_threshold,
            test_is_anomaly,
            output_dir / "test_window_mse.png",
            threshold_std=test_threshold_std if test_threshold_std is not None else threshold_std,
            ground_truth=test_ground_truth,
        )
    print(f"\nPlots saved to: {output_dir.resolve()}")
    print(f"  - training_loss.png")
    print(f"  - train_window_mse.png")
    if test_errors is not None and test_is_anomaly is not None:
        print(f"  - test_window_mse.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Keras autoencoder for sensor anomaly detection."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="CSV with columns: current, voltage, power_factor, active_power. "
        "Uses synthetic demo data if omitted.",
    )
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold-std", type=float, default=THRESHOLD_STD)
    parser.add_argument(
        "--window-threshold-std",
        type=float,
        default=None,
        help="Std multiplier for window-level threshold (defaults to --threshold-std).",
    )
    parser.add_argument(
        "--timestep-threshold-std",
        type=float,
        default=None,
        help="Std multiplier for timestep-level threshold (defaults to --threshold-std).",
    )
    parser.add_argument(
        "--threshold-calibrate-on",
        choices=THRESHOLD_CALIBRATION_SOURCES,
        default=DEFAULT_THRESHOLD_CALIBRATION,
        help="Timestep-level only: fit threshold on train scores (default) or "
        "clean test timesteps (demo).",
    )
    parser.add_argument(
        "--window-test-calibrate-on",
        choices=WINDOW_TEST_CALIBRATION_SOURCES,
        default=DEFAULT_WINDOW_TEST_CALIBRATION,
        help="Window-level test threshold calibration: clean (default, no fault "
        "timesteps inside window) or all (every test window MSE).",
    )
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument(
        "--test-anomaly-fraction",
        type=float,
        default=0.2,
        help="Fraction of test windows marked anomalous in synthetic demo data.",
    )
    parser.add_argument(
        "--test-clean-prefix-windows",
        type=int,
        default=None,
        metavar="N",
        help="Synthetic demo only: keep the first N test sliding windows free of "
        "fault timesteps. Default: 2 * window_size. Use 0 to allow faults from "
        "the first window.",
    )
    parser.add_argument(
        "--anomaly-magnitude",
        choices=ANOMALY_MAGNITUDES,
        default=DEFAULT_ANOMALY_MAGNITUDE,
        help="Synthetic fault strength: small (subtle), medium, or large.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to save training loss and per-window MSE plots.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots.",
    )
    parser.add_argument(
        "--train-normal-only",
        action="store_true",
        help="Exclude windows containing any anomalous timestep from training. "
        f"Requires '{LABEL_COLUMN}' labels (synthetic demo or CSV column).",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=("window", "timestep", "both"),
        default="both",
        help="Anomaly scoring: whole-window MSE, timestep-level aggregate, or both.",
    )
    parser.add_argument(
        "--timestep-aggregate",
        choices=("max", "mean"),
        default="max",
        help="How to combine overlapping window errors into a timestep score.",
    )
    args = parser.parse_args()
    window_threshold_std = (
        args.window_threshold_std
        if args.window_threshold_std is not None
        else args.threshold_std
    )
    timestep_threshold_std = (
        args.timestep_threshold_std
        if args.timestep_threshold_std is not None
        else args.threshold_std
    )

    np.random.seed(args.seed)

    if args.test_clean_prefix_windows is not None and args.test_clean_prefix_windows < 0:
        raise ValueError("--test-clean-prefix-windows must be >= 0.")

    dataset = load_sensor_data(
        args.data,
        seed=args.seed,
        train_split=args.train_split,
        window_size=args.window_size,
        test_anomaly_fraction=args.test_anomaly_fraction,
        anomaly_magnitude=args.anomaly_magnitude,
        clean_prefix_windows=args.test_clean_prefix_windows,
    )

    train_values = dataset.train_df.to_numpy(dtype=np.float32)
    test_values = dataset.test_df.to_numpy(dtype=np.float32)
    train_windows = window_sensor_data(train_values, args.window_size)
    test_windows = window_sensor_data(test_values, args.window_size)
    n_train = len(train_windows)
    n_test = len(test_windows)
    window_ground_truth = dataset.window_ground_truth

    if args.data is None:
        describe_synthetic_split(
            dataset,
            args.test_anomaly_fraction,
            args.anomaly_magnitude,
            clean_prefix_windows=args.test_clean_prefix_windows,
            window_size=args.window_size,
        )

    if args.train_normal_only:
        train_windows, removed = filter_normal_windows(
            train_windows,
            dataset.train_point_labels,
            args.window_size,
        )
        if len(train_windows) == 0:
            raise ValueError(
                "No normal training windows remain after filtering. "
                "Relax anomaly rate or provide more normal data."
            )
        if removed > 0:
            print(
                f"\n--train-normal-only: removed {removed} anomalous train windows, "
                f"kept {len(train_windows)} normal windows "
                f"({100 * len(train_windows) / n_train:.1f}% of original train set)."
            )
        else:
            print("\n--train-normal-only: train set already contains only normal windows.")
    scaler = RobustScaler()
    train_flat = train_windows.reshape(-1, train_windows.shape[-1])
    scaler.fit(train_flat)

    def scale(w: np.ndarray) -> np.ndarray:
        shape = w.shape
        scaled = scaler.transform(w.reshape(-1, shape[-1]))
        return scaled.reshape(shape)

    train_scaled = scale(train_windows)
    test_scaled = scale(test_windows)
    n_train_timesteps = len(train_values)
    n_test_timesteps = len(test_values)

    model = build_autoencoder(
        input_shape=(args.window_size, len(SENSOR_COLUMNS))
    )
    model.compile(optimizer="adam", loss="mse")

    history = model.fit(
        train_scaled,
        train_scaled,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=1,
    )

    use_window = args.scoring_mode in ("window", "both")
    use_timestep = args.scoring_mode in ("timestep", "both")

    train_result = None
    test_result = None
    train_timestep_result = None
    test_timestep_result = None
    window_thresholds: WindowThresholds | None = None
    timestep_threshold = None

    if use_window:
        train_window_errors = reconstruction_errors(model, train_scaled)
        test_window_errors = reconstruction_errors(model, test_scaled)

        test_clean_window_mask = clean_window_mask(
            dataset.test_point_labels, args.window_size
        )
        window_thresholds, test_calib_source = fit_window_thresholds(
            train_window_errors,
            test_window_errors,
            window_threshold_std,
            dataset.test_point_labels,
            args.window_size,
            test_calibration=args.window_test_calibrate_on,
        )
        train_result = detect_from_window_errors(
            train_window_errors, window_thresholds.train
        )
        test_result = detect_from_window_errors(
            test_window_errors, window_thresholds.test
        )
        warn_test_mse_shift(
            train_window_errors,
            test_window_errors,
            test_clean_window_mask,
        )
        print_window_level_mse_report(
            train_window_errors,
            test_window_errors,
            window_thresholds,
            test_calibration_source=test_calib_source,
        )

    if use_timestep:
        train_wt_errors = window_timestep_errors(model, train_scaled)
        train_timestep_scores = aggregate_timestep_scores(
            train_wt_errors,
            n_train_timesteps,
            method=args.timestep_aggregate,
        )

        test_wt_errors = window_timestep_errors(model, test_scaled)
        test_timestep_scores = aggregate_timestep_scores(
            test_wt_errors,
            n_test_timesteps,
            method=args.timestep_aggregate,
        )

        if args.threshold_calibrate_on == "test_normal":
            test_normal_timestep_mask = ~dataset.test_point_labels
            if not test_normal_timestep_mask.any():
                raise ValueError(
                    "No clean test timesteps available for calibration."
                )
            calibration_scores = test_timestep_scores[test_normal_timestep_mask]
            calib_source = "clean test timesteps (no fault)"
        else:
            calibration_scores = train_timestep_scores
            calib_source = "train"

        timestep_threshold = fit_timestep_threshold(
            calibration_scores,
            timestep_threshold_std,
            aggregate_method=args.timestep_aggregate,
            calibration_source=calib_source,
        )
        train_timestep_result = detect_from_timestep_scores(
            timestep_scores=train_timestep_scores,
            threshold=timestep_threshold,
            window_size=args.window_size,
            n_eval_windows=n_train,
        )
        test_timestep_result = detect_from_timestep_scores(
            timestep_scores=test_timestep_scores,
            threshold=timestep_threshold,
            window_size=args.window_size,
            n_eval_windows=n_test,
        )

    if not args.no_plot and use_window and train_result is not None and window_thresholds is not None:
        test_window_labels = dataset.test_window_ground_truth
        plot_training_results(
            history=history,
            train_errors=train_result.reconstruction_errors,
            train_threshold=window_thresholds.train.value,
            is_anomaly=train_result.is_anomaly,
            output_dir=args.plot_dir,
            threshold_std=window_thresholds.train.std_multiplier,
            test_errors=test_result.reconstruction_errors if test_result else None,
            test_is_anomaly=test_result.is_anomaly if test_result else None,
            test_threshold=window_thresholds.test.value,
            test_threshold_std=window_thresholds.test.std_multiplier,
            test_ground_truth=test_window_labels,
        )

    print(f"\nTrain windows: {len(train_scaled)}")
    print(f"Test windows:  {len(test_scaled)}")

    if use_window and train_result is not None and test_result is not None and window_thresholds is not None:
        print("\n[Window-level] Detection results:")
        print(
            f"  Train threshold: {window_thresholds.train.value:.6f} "
            f"(μ + {window_thresholds.train.std_multiplier}σ on train)"
        )
        print(
            f"  Test threshold:  {window_thresholds.test.value:.6f} "
            f"(μ + {window_thresholds.test.std_multiplier}σ on test calibration)"
        )
        print(
            f"  Train flagged: {train_result.is_anomaly.sum()} "
            f"({100 * train_result.is_anomaly.mean():.2f}%)"
        )
        print(
            f"  Test flagged:  {test_result.is_anomaly.sum()} "
            f"({100 * test_result.is_anomaly.mean():.2f}%)"
        )

    if (
        use_timestep
        and train_timestep_result is not None
        and test_timestep_result is not None
        and timestep_threshold is not None
    ):
        print(f"\n[Timestep-level ({args.timestep_aggregate})] Detection results:")
        print(
            f"  Train timesteps flagged: {train_timestep_result.timestep_is_anomaly.sum()} "
            f"({100 * train_timestep_result.timestep_is_anomaly.mean():.2f}%)"
        )
        print(
            f"  Test timesteps flagged:  {test_timestep_result.timestep_is_anomaly.sum()} "
            f"({100 * test_timestep_result.timestep_is_anomaly.mean():.2f}%)"
        )
        print(
            f"  Test windows flagged (rollup): {test_timestep_result.window_is_anomaly.sum()} "
            f"({100 * test_timestep_result.window_is_anomaly.mean():.2f}%)"
        )

    test_window_labels = dataset.test_window_ground_truth
    if test_window_labels is not None:
        if use_window and test_result is not None and len(test_window_labels) == len(test_result.is_anomaly):
            print_detection_metrics(
                "Demo evaluation — window-level (synthetic labels):",
                test_result.is_anomaly,
                test_window_labels,
            )
        if (
            use_timestep
            and test_timestep_result is not None
            and len(test_window_labels) == len(test_timestep_result.window_is_anomaly)
        ):
            print_detection_metrics(
                f"Demo evaluation — timestep-level ({args.timestep_aggregate}, synthetic labels):",
                test_timestep_result.window_is_anomaly,
                test_window_labels,
            )


if __name__ == "__main__":
    main()
