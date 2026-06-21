import numpy as np
import pytest
from core.membership import build_mf, MEMBERSHIP_REGISTRY


def mf(mf_type, params):
    return build_mf({"mf_type": mf_type, "params": params})


# ── Реєстр ───────────────────────────────────────────────────────────────────

def test_registry_has_thirteen_types():
    expected = {"trimf", "trapmf", "gaussmf",
                "gauss2mf", "gbellmf", "sigmf", "dsigmf", "psigmf",
                "zmf", "smf", "pimf", "linsmf", "linzmf"}
    assert expected == set(MEMBERSHIP_REGISTRY.keys())


# ── gauss2mf ─────────────────────────────────────────────────────────────────

def test_gauss2mf_plateau_is_one():
    f = mf("gauss2mf", [0.1, 1.0, 0.1, 3.0])
    assert abs(f.value(np.array([2.0]))[0] - 1.0) < 1e-9

def test_gauss2mf_tails_near_zero():
    f = mf("gauss2mf", [0.1, 1.0, 0.1, 3.0])
    assert f.value(np.array([-2.0]))[0] < 0.01
    assert f.value(np.array([7.0]))[0] < 0.01

def test_gauss2mf_symmetry_equal_sigmas():
    f = mf("gauss2mf", [0.5, 0.0, 0.5, 2.0])
    # Симетричний відносно середини плато (1.0)
    assert abs(f.value(np.array([-0.5]))[0] - f.value(np.array([2.5]))[0]) < 1e-9


# ── gbellmf ───────────────────────────────────────────────────────────────────

def test_gbellmf_peak_at_c():
    f = mf("gbellmf", [1.0, 2.0, 5.0])
    assert abs(f.value(np.array([5.0]))[0] - 1.0) < 1e-9

def test_gbellmf_symmetry():
    f = mf("gbellmf", [1.0, 2.0, 0.0])
    assert abs(f.value(np.array([1.0]))[0] - f.value(np.array([-1.0]))[0]) < 1e-9

def test_gbellmf_decreasing_far_from_peak():
    f = mf("gbellmf", [1.0, 2.0, 0.0])
    assert f.value(np.array([0.0]))[0] > f.value(np.array([2.0]))[0]


# ── sigmf ─────────────────────────────────────────────────────────────────────

def test_sigmf_at_inflection_is_half():
    f = mf("sigmf", [10.0, 5.0])
    assert abs(f.value(np.array([5.0]))[0] - 0.5) < 1e-6

def test_sigmf_positive_a_increases():
    f = mf("sigmf", [10.0, 5.0])
    assert f.value(np.array([4.0]))[0] < f.value(np.array([6.0]))[0]

def test_sigmf_negative_a_decreases():
    f = mf("sigmf", [-10.0, 5.0])
    assert f.value(np.array([4.0]))[0] > f.value(np.array([6.0]))[0]


# ── dsigmf ────────────────────────────────────────────────────────────────────
# Параметри [100, 2, 100, 8]: плато між x=2 і x=8, обидві сигмоїди з a=100

def test_dsigmf_peak_near_one():
    """Різниця двох сигмоїд, середня ділянка ≈ 1."""
    f = mf("dsigmf", [100.0, 2.0, 100.0, 8.0])
    assert f.value(np.array([5.0]))[0] > 0.99

def test_dsigmf_outside_near_zero():
    f = mf("dsigmf", [100.0, 2.0, 100.0, 8.0])
    assert f.value(np.array([0.0]))[0] < 0.01
    assert f.value(np.array([12.0]))[0] < 0.01

def test_dsigmf_nonnegative():
    f = mf("dsigmf", [100.0, 2.0, 100.0, 8.0])
    fs = f.discretize()
    assert np.all(fs.mu >= 0.0)


# ── psigmf ────────────────────────────────────────────────────────────────────
# Параметри [100, 2, -100, 8]: a1>0 зростає ліворуч, a2<0 спадає праворуч

def test_psigmf_peak_near_one():
    """Добуток двох сигмоїд: a2<0 дає спадну на правій межі."""
    f = mf("psigmf", [100.0, 2.0, -100.0, 8.0])
    assert f.value(np.array([5.0]))[0] > 0.95

def test_psigmf_outside_near_zero():
    f = mf("psigmf", [100.0, 2.0, -100.0, 8.0])
    assert f.value(np.array([0.0]))[0] < 0.05
    assert f.value(np.array([12.0]))[0] < 0.05

def test_psigmf_nonnegative():
    f = mf("psigmf", [100.0, 2.0, -100.0, 8.0])
    fs = f.discretize()
    assert np.all(fs.mu >= 0.0)


# ── zmf ───────────────────────────────────────────────────────────────────────

def test_zmf_one_at_left():
    f = mf("zmf", [0.0, 1.0])
    assert abs(f.value(np.array([0.0]))[0] - 1.0) < 1e-9

def test_zmf_zero_at_right():
    f = mf("zmf", [0.0, 1.0])
    assert abs(f.value(np.array([1.0]))[0]) < 1e-9

def test_zmf_monotone_decreasing():
    f = mf("zmf", [0.0, 1.0])
    xs = np.linspace(0.0, 1.0, 20)
    vals = f.value(xs)
    assert np.all(np.diff(vals) <= 1e-9)


# ── smf ───────────────────────────────────────────────────────────────────────

def test_smf_zero_at_left():
    f = mf("smf", [0.0, 1.0])
    assert abs(f.value(np.array([0.0]))[0]) < 1e-9

def test_smf_one_at_right():
    f = mf("smf", [0.0, 1.0])
    assert abs(f.value(np.array([1.0]))[0] - 1.0) < 1e-9

def test_smf_monotone_increasing():
    f = mf("smf", [0.0, 1.0])
    xs = np.linspace(0.0, 1.0, 20)
    vals = f.value(xs)
    assert np.all(np.diff(vals) >= -1e-9)


# ── pimf ──────────────────────────────────────────────────────────────────────

def test_pimf_plateau_is_one():
    f = mf("pimf", [0.0, 0.3, 0.7, 1.0])
    assert abs(f.value(np.array([0.5]))[0] - 1.0) < 1e-9

def test_pimf_zero_outside():
    f = mf("pimf", [0.0, 0.3, 0.7, 1.0])
    assert abs(f.value(np.array([-0.1]))[0]) < 1e-9
    assert abs(f.value(np.array([1.1]))[0]) < 1e-9

def test_pimf_symmetric():
    f = mf("pimf", [0.0, 0.25, 0.75, 1.0])
    assert abs(f.value(np.array([0.1]))[0] - f.value(np.array([0.9]))[0]) < 1e-9


# ── linsmf ────────────────────────────────────────────────────────────────────

def test_linsmf_zero_at_a():
    f = mf("linsmf", [0.994, 1.0])
    assert abs(f.value(np.array([0.994]))[0]) < 1e-9

def test_linsmf_one_at_b():
    f = mf("linsmf", [0.994, 1.0])
    assert abs(f.value(np.array([1.0]))[0] - 1.0) < 1e-9

def test_linsmf_linear_inside():
    f = mf("linsmf", [0.0, 1.0])
    assert abs(f.value(np.array([0.5]))[0] - 0.5) < 1e-9


# ── linzmf ────────────────────────────────────────────────────────────────────

def test_linzmf_one_at_a():
    f = mf("linzmf", [0.994, 1.0])
    assert abs(f.value(np.array([0.994]))[0] - 1.0) < 1e-9

def test_linzmf_zero_at_b():
    f = mf("linzmf", [0.994, 1.0])
    assert abs(f.value(np.array([1.0]))[0]) < 1e-9

def test_linzmf_linear_inside():
    f = mf("linzmf", [0.0, 1.0])
    assert abs(f.value(np.array([0.5]))[0] - 0.5) < 1e-9