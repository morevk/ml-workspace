import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing the script as a module when running pytest from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoencoder_anomaly_detection import (
    TEST_WINDOW_CALIBRATION_ALL,
    TEST_WINDOW_CALIBRATION_CLEAN,
    WINDOW_SIZE,
    clean_window_mask,
    generate_synthetic_dataset,
    generate_test_data,
    generate_train_data,
    synthetic_window_counts,
    select_test_window_calibration_errors,
)


def _assert_frames_close(a: pd.DataFrame, b: pd.DataFrame) -> None:
    assert list(a.columns) == list(b.columns)
    assert a.shape == b.shape
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=0.0, atol=0.0)


def test_isolated_test_reproducible_and_matches_manual_build() -> None:
    """
    Test data is generated from a fresh default_rng(seed), not from train's
    remaining RNG state. Same seed must reproduce the same test split.
    """
    seed = 123
    n_samples = 5000
    window_size = 50
    train_split = 0.8
    test_anomaly_fraction = 0.2
    anomaly_magnitude = "small"

    n_train_windows, n_test_windows = synthetic_window_counts(
        n_samples=n_samples,
        window_size=window_size,
        train_split=train_split,
    )

    ds = generate_synthetic_dataset(
        n_samples=n_samples,
        seed=seed,
        train_split=train_split,
        window_size=window_size,
        test_anomaly_fraction=test_anomaly_fraction,
        anomaly_magnitude=anomaly_magnitude,
    )

    ds_repeat = generate_synthetic_dataset(
        n_samples=n_samples,
        seed=seed,
        train_split=train_split,
        window_size=window_size,
        test_anomaly_fraction=test_anomaly_fraction,
        anomaly_magnitude=anomaly_magnitude,
    )

    _assert_frames_close(ds.test_df, ds_repeat.test_df)
    np.testing.assert_array_equal(ds.test_point_labels, ds_repeat.test_point_labels)
    np.testing.assert_array_equal(
        ds.test_window_ground_truth, ds_repeat.test_window_ground_truth
    )

    train_rng = np.random.default_rng(seed)
    train_df, train_labels, train_window_gt = generate_train_data(
        n_train_windows=n_train_windows,
        window_size=window_size,
        rng=train_rng,
    )
    test_timesteps = n_test_windows + window_size - 1
    test_baseline = train_df.iloc[:test_timesteps].reset_index(drop=True)
    test_rng = np.random.default_rng(seed)
    test_df, test_labels, test_window_gt = generate_test_data(
        n_test_windows=n_test_windows,
        window_size=window_size,
        test_anomaly_fraction=test_anomaly_fraction,
        anomaly_magnitude=anomaly_magnitude,
        rng=test_rng,
        normal_baseline=test_baseline,
    )

    _assert_frames_close(ds.train_df, train_df)
    _assert_frames_close(ds.test_df, test_df)
    np.testing.assert_array_equal(ds.train_point_labels, train_labels)
    np.testing.assert_array_equal(ds.test_point_labels, test_labels)
    np.testing.assert_array_equal(ds.train_window_ground_truth, train_window_gt)
    np.testing.assert_array_equal(ds.test_window_ground_truth, test_window_gt)


def test_isolated_test_normal_timesteps_match_train_prefix() -> None:
    """
    Test baseline is copied from the train prefix; unperturbed timesteps must
    match that prefix exactly (fault injection only on selected indices).
    """
    ds = generate_synthetic_dataset(seed=42, anomaly_magnitude="small")
    n_test = len(ds.test_df)
    train_prefix = ds.train_df.iloc[:n_test]
    normal_mask = ~ds.test_point_labels

    assert normal_mask.any(), "expected some normal timesteps in test trace"
    np.testing.assert_allclose(
        train_prefix.to_numpy()[normal_mask],
        ds.test_df.to_numpy()[normal_mask],
        rtol=0.0,
        atol=0.0,
    )


def test_initial_test_windows_are_clean_sliding_windows() -> None:
    """First 2*window_size test windows must contain no fault timesteps."""
    ds = generate_synthetic_dataset(seed=42, anomaly_magnitude="small")
    clean = clean_window_mask(ds.test_point_labels, WINDOW_SIZE)
    n_clean_prefix = 2 * WINDOW_SIZE
    assert clean[:n_clean_prefix].all(), (
        f"expected windows 0..{n_clean_prefix - 1} to be fully clean"
    )


def test_initial_test_windows_match_train() -> None:
    """Clean prefix test windows must match the same train windows exactly."""
    ds = generate_synthetic_dataset(seed=42, anomaly_magnitude="small")
    n_test = len(ds.test_df)
    train_w = ds.train_df.iloc[:n_test].to_numpy()
    test_w = ds.test_df.to_numpy()
    clean = clean_window_mask(ds.test_point_labels, WINDOW_SIZE)
    for i in range(len(clean)):
        if not clean[i]:
            continue
        start, end = i, i + WINDOW_SIZE
        np.testing.assert_allclose(
            train_w[start:end], test_w[start:end], rtol=0.0, atol=0.0
        )


def test_window_calibration_all_uses_every_test_window() -> None:
    ds = generate_synthetic_dataset(seed=42, anomaly_magnitude="small")
    errors = np.arange(len(ds.test_window_ground_truth), dtype=float)
    calib, label = select_test_window_calibration_errors(
        errors,
        ds.test_point_labels,
        WINDOW_SIZE,
        "all",
    )
    assert label == TEST_WINDOW_CALIBRATION_ALL
    np.testing.assert_array_equal(calib, errors)


def test_window_calibration_clean_subset() -> None:
    ds = generate_synthetic_dataset(seed=42, anomaly_magnitude="small")
    errors = np.arange(len(ds.test_window_ground_truth), dtype=float)
    calib, label = select_test_window_calibration_errors(
        errors,
        ds.test_point_labels,
        WINDOW_SIZE,
        "clean",
    )
    assert label == TEST_WINDOW_CALIBRATION_CLEAN
    clean = clean_window_mask(ds.test_point_labels, WINDOW_SIZE)
    np.testing.assert_array_equal(calib, errors[clean])


def test_train_test_have_similar_normal_statistics() -> None:
    """
    Train and test are generated from the same underlying process, so normal
    channel means/stds should be in the same ballpark (not orders of magnitude).
    This does NOT guarantee reconstruction MSE alignment.
    """
    ds = generate_synthetic_dataset(seed=42, anomaly_magnitude="small")

    train = ds.train_df
    test = ds.test_df

    train_mu = train.mean()
    test_mu = test.mean()
    train_std = train.std()
    test_std = test.std()

    # Loose tolerances: just prevent extreme mismatches (e.g. 1e9 vs 1e3).
    ratio_mu = (test_mu.abs() + 1e-9) / (train_mu.abs() + 1e-9)
    ratio_std = (test_std.abs() + 1e-9) / (train_std.abs() + 1e-9)

    assert float(ratio_mu.max()) < 5.0
    assert float(ratio_std.max()) < 5.0
