import numpy as np
import pytest
from core.membership import build_mf
from core.ptimely import (
    PtimelyCurve,
    compute_ptimely,
    ptimely_analytical_trimf,
    ptimely_analytical_trapmf,
    ptimely_possibility,
    ptimely_integral,
    select_default_method,
    extract_trimf_params_from_fuzzyset,
    extract_trapmf_params_from_fuzzyset,
)
from core.fuzzy_arith import FuzzySet, fuzzy_add


def test_select_method_for_mixed():
    """Mixed/gaussian/None → possibility; trimf/trapmf → analytical."""
    assert select_default_method({"mf_type": "mixed"})   == "possibility"
    assert select_default_method({"mf_type": "gaussmf"}) == "possibility"
    assert select_default_method({"mf_type": "trimf"})   == "analytical"
    assert select_default_method({"mf_type": "trapmf"})  == "analytical"
    assert select_default_method(None)                    == "possibility"


# ── Метод 1: аналітичний ─────────────────────────────────────────────────────

def test_analytical_trimf_below_left():
    """T_дир ≤ a → 0."""
    grid = np.linspace(0, 30, 200)
    p = ptimely_analytical_trimf(10, 15, 20, grid)
    assert p[grid <= 10].max() < 1e-9


def test_analytical_trimf_above_mode():
    """T_дир > b (мода) → 1."""
    grid = np.linspace(0, 30, 200)
    p = ptimely_analytical_trimf(10, 15, 20, grid)
    assert (p[grid > 15] >= 1.0 - 1e-9).all()


def test_analytical_trimf_at_mode():
    """У точці b значення = 1."""
    p = ptimely_analytical_trimf(10, 15, 20, np.array([15.0]))
    assert abs(p[0] - 1.0) < 1e-9


def test_analytical_trimf_midway_rise():
    """На середині висхідної гілки значення = 0.5."""
    p = ptimely_analytical_trimf(10, 15, 20, np.array([12.5]))
    assert abs(p[0] - 0.5) < 1e-9


def test_analytical_trapmf_plateau():
    """T_дир у плато (> b) → 1."""
    p = ptimely_analytical_trapmf(10, 12, 15, 18, np.array([14.0]))
    assert abs(p[0] - 1.0) < 1e-9


# ── Метод 2: можливісний ─────────────────────────────────────────────────────

def test_possibility_below_support():
    """T_дир нижче носія MT → 0."""
    mt = build_mf({"mf_type": "trimf", "params": [10, 15, 20]}).discretize()
    p = ptimely_possibility(mt, np.array([5.0]))
    assert p[0] < 1e-9


def test_possibility_above_support():
    """T_дир вище носія → 1 (μ_T досягає 1 усередині носія)."""
    mt = build_mf({"mf_type": "trimf", "params": [10, 15, 20]}).discretize()
    p = ptimely_possibility(mt, np.array([25.0]))
    assert abs(p[0] - 1.0) < 1e-3


def test_possibility_monotonically_increasing():
    """P_св(T_дир) для possibility-методу має бути неспадною функцією."""
    mt = build_mf({"mf_type": "gaussmf", "params": [2.0, 15.0]}).discretize()
    grid = np.linspace(0, 30, 100)
    p = ptimely_possibility(mt, grid)
    assert (np.diff(p) >= -1e-9).all()


def test_possibility_at_peak_equals_one():
    """У точці моди (де μ=1) P_св досягає 1."""
    mt = build_mf({"mf_type": "trimf", "params": [10, 15, 20]}).discretize()
    p = ptimely_possibility(mt, np.array([15.0]))
    assert abs(p[0] - 1.0) < 1e-3


# ── Метод 3: інтегральний ────────────────────────────────────────────────────

def test_integral_below_support_zero():
    mt = build_mf({"mf_type": "trimf", "params": [10, 15, 20]}).discretize()
    p = ptimely_integral(mt, np.array([5.0]))
    assert p[0] < 1e-3


def test_integral_above_support_one():
    mt = build_mf({"mf_type": "trimf", "params": [10, 15, 20]}).discretize()
    p = ptimely_integral(mt, np.array([25.0]))
    assert abs(p[0] - 1.0) < 1e-3


def test_integral_symmetric_at_center_is_half():
    """Для симетричної ФП у точці центру P_св ≈ 0.5."""
    mt = build_mf({"mf_type": "trimf", "params": [10, 15, 20]}).discretize()
    p = ptimely_integral(mt, np.array([15.0]))
    assert abs(p[0] - 0.5) < 0.05


def test_integral_monotonic():
    mt = build_mf({"mf_type": "gaussmf", "params": [2.0, 15.0]}).discretize()
    grid = np.linspace(0, 30, 100)
    p = ptimely_integral(mt, grid)
    assert (np.diff(p) >= -1e-9).all()


def test_possibility_smooth_for_gaussmf():
    """Possibility-крива для gaussmf — монотонна і без різких стрибків > 0.05."""
    mt = build_mf({"mf_type": "gaussmf", "params": [2.0, 15.0]}).discretize()
    grid = np.linspace(0, 30, 500)
    p = ptimely_possibility(mt, grid)
    diffs = np.diff(p)
    assert (diffs >= -1e-9).all(), "Has to be monotonically non-decreasing"
    assert diffs.max() < 0.05, f"Max jump {diffs.max():.4f} > 0.05"


# ── Універсальний фасад ───────────────────────────────────────────────────────

def test_compute_returns_curve():
    mt = build_mf({"mf_type": "trimf", "params": [10, 15, 20]}).discretize()
    curve = compute_ptimely(mt, t_dir_max=30, n_points=100, method="possibility")
    assert isinstance(curve, PtimelyCurve)
    assert len(curve.t_dir) == 100
    assert len(curve.p_sv) == 100
    assert curve.method == "possibility"


def test_compute_auto_for_trimf_uses_analytical():
    mt_spec = {"mf_type": "trimf", "params": [10, 15, 20]}
    mt = build_mf(mt_spec).discretize()
    curve = compute_ptimely(mt, t_dir_max=30, method="auto", mt_spec=mt_spec)
    assert curve.method == "analytical"


def test_compute_auto_for_gauss_uses_possibility():
    mt_spec = {"mf_type": "gaussmf", "params": [2.0, 15.0]}
    mt = build_mf(mt_spec).discretize()
    curve = compute_ptimely(mt, t_dir_max=30, method="auto", mt_spec=mt_spec)
    assert curve.method == "possibility"


def test_at_method_interpolates():
    mt = build_mf({"mf_type": "trimf", "params": [10, 15, 20]}).discretize()
    curve = compute_ptimely(mt, t_dir_max=30, method="possibility")
    assert abs(curve.at(15.0) - 1.0) < 0.05
    assert curve.at(5.0) < 0.05


# ── Узгодженість трьох методів ────────────────────────────────────────────────

def test_all_three_methods_zero_below_support():
    """Усі 3 методи дають 0 при T_дир значно нижче носія."""
    mt_spec = {"mf_type": "trimf", "params": [10, 15, 20]}
    mt = build_mf(mt_spec).discretize()
    assert ptimely_analytical_trimf(10, 15, 20, np.array([5.0]))[0] < 1e-3
    assert ptimely_possibility(mt, np.array([5.0]))[0] < 1e-3
    assert ptimely_integral(mt, np.array([5.0]))[0] < 1e-3


def test_all_three_methods_one_well_above_support():
    """Усі 3 методи дають 1 при T_дир значно вище носія."""
    mt_spec = {"mf_type": "trimf", "params": [10, 15, 20]}
    mt = build_mf(mt_spec).discretize()
    assert abs(ptimely_analytical_trimf(10, 15, 20, np.array([25.0]))[0] - 1.0) < 1e-3
    assert abs(ptimely_possibility(mt, np.array([25.0]))[0] - 1.0) < 1e-3
    assert abs(ptimely_integral(mt, np.array([25.0]))[0] - 1.0) < 1e-3


# ── Вилучення параметрів з результуючої ФП ───────────────────────────────────

def test_extract_trimf_from_sum_of_two_trimfs():
    """fuzzy_add(trimf[10,12,14], trimf[8,10,12]) ≈ trimf[18,22,26].
    Витягнуті параметри мають бути в межах ±1 від очікуваних.
    """
    fs1 = build_mf({"mf_type": "trimf", "params": [10, 12, 14]}).discretize()
    fs2 = build_mf({"mf_type": "trimf", "params": [8, 10, 12]}).discretize()
    result = fuzzy_add(fs1, fs2)
    a, b, c = extract_trimf_params_from_fuzzyset(result)
    assert abs(a - 18) < 1, f"a={a:.3f}, expected ≈18"
    assert abs(b - 22) < 1, f"b={b:.3f}, expected ≈22"
    assert abs(c - 26) < 1, f"c={c:.3f}, expected ≈26"


def test_extract_trimf_raises_for_bimodal():
    """Бімодальна ФП → ValueError."""
    x = np.linspace(0, 10, 201)
    mu = np.zeros(201)
    # Два окремих горби
    mu[30:70] = np.linspace(0, 1, 40)
    mu[70:110] = np.linspace(1, 0, 40)
    mu[130:170] = np.linspace(0, 1, 40)
    mu[170:201] = np.linspace(1, 0, 31)
    fs = FuzzySet(x=x, mu=mu)
    with pytest.raises(ValueError, match="not approximately triangular"):
        extract_trimf_params_from_fuzzyset(fs)


def test_extract_trapmf_from_sum_of_two_trapmfs():
    """fuzzy_add(trapmf[2,4,6,8], trapmf[1,3,5,7]) ≈ trapmf[3,7,11,15].
    Плато результату — між 7 і 11.
    """
    fs1 = build_mf({"mf_type": "trapmf", "params": [2, 4, 6, 8]}).discretize()
    fs2 = build_mf({"mf_type": "trapmf", "params": [1, 3, 5, 7]}).discretize()
    result = fuzzy_add(fs1, fs2)
    a, b, c, d = extract_trapmf_params_from_fuzzyset(result)
    assert abs(a - 3)  < 1, f"a={a:.3f}"
    assert abs(b - 7)  < 1, f"b={b:.3f}"
    assert abs(c - 11) < 1, f"c={c:.3f}"
    assert abs(d - 15) < 1, f"d={d:.3f}"


def test_extract_trapmf_raises_for_narrow_plateau():
    """Трикутна ФП (плато < 3 точок) → ValueError з extract_trapmf."""
    fs = build_mf({"mf_type": "trimf", "params": [10, 15, 20]}).discretize()
    with pytest.raises(ValueError, match="too narrow"):
        extract_trapmf_params_from_fuzzyset(fs)