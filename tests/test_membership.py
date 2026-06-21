import numpy as np
import pytest
from core.membership import build_mf, MEMBERSHIP_REGISTRY


def test_registry_has_three_types():
    assert "trimf" in MEMBERSHIP_REGISTRY
    assert "trapmf" in MEMBERSHIP_REGISTRY
    assert "gaussmf" in MEMBERSHIP_REGISTRY


def test_trimf_peak_is_one():
    mf = build_mf({"mf_type": "trimf", "params": [0.0, 0.5, 1.0]})
    assert abs(mf.value(np.array([0.5]))[0] - 1.0) < 1e-9


def test_trimf_outside_is_zero():
    mf = build_mf({"mf_type": "trimf", "params": [0.0, 0.5, 1.0]})
    assert mf.value(np.array([-0.1, 1.1]))[0] == 0
    assert mf.value(np.array([-0.1, 1.1]))[1] == 0


def test_trimf_degenerate():
    """Вироджена трикутна (a=b=c) — дельта-функція."""
    mf = build_mf({"mf_type": "trimf", "params": [5.0, 5.0, 5.0]})
    fs = mf.discretize(n_points=11)
    peak_idx = np.argmin(np.abs(fs.x - 5.0))
    assert fs.mu[peak_idx] == 1.0


def test_trapmf_plateau_is_one():
    mf = build_mf({"mf_type": "trapmf", "params": [0.0, 0.3, 0.7, 1.0]})
    vals = mf.value(np.array([0.3, 0.5, 0.7]))
    assert all(abs(v - 1.0) < 1e-9 for v in vals)


def test_gaussmf_peak_at_c():
    mf = build_mf({"mf_type": "gaussmf", "params": [1.0, 5.0]})
    assert abs(mf.value(np.array([5.0]))[0] - 1.0) < 1e-9


def test_gaussmf_symmetry():
    mf = build_mf({"mf_type": "gaussmf", "params": [1.0, 0.0]})
    assert abs(mf.value(np.array([1.0]))[0] - mf.value(np.array([-1.0]))[0]) < 1e-9
