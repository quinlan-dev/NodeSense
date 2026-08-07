"""Tests for the log+standard-scaling preprocessing transform in train.py."""

import numpy as np

from data import FEATURE_NAMES, generate_sessions
from train import preprocess_apply, preprocess_fit


def test_preprocess_preserves_shape():
    X, _, _ = generate_sessions(n_sessions=50, seed=1)
    scaler = preprocess_fit(X)
    Xs = preprocess_apply(X, scaler)
    assert Xs.shape == X.shape


def test_preprocess_output_is_finite():
    X, _, _ = generate_sessions(n_sessions=50, seed=1)
    scaler = preprocess_fit(X)
    Xs = preprocess_apply(X, scaler)
    assert np.isfinite(Xs).all()


def test_preprocess_standardizes_the_fit_set():
    # Applying the scaler to the exact data it was fit on must reproduce
    # standard scaling: per-feature mean ~0, std ~1.
    X, _, _ = generate_sessions(n_sessions=400, seed=2)
    scaler = preprocess_fit(X)
    Xs = preprocess_apply(X, scaler)
    flat = Xs.reshape(-1, len(FEATURE_NAMES))
    assert np.allclose(flat.mean(axis=0), 0, atol=1e-3)
    assert np.allclose(flat.std(axis=0), 1, atol=1e-2)


def test_preprocess_is_deterministic():
    X, _, _ = generate_sessions(n_sessions=30, seed=3)
    scaler = preprocess_fit(X)
    a = preprocess_apply(X, scaler)
    b = preprocess_apply(X, scaler)
    assert np.array_equal(a, b)
