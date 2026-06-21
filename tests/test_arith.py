import numpy as np
from core.membership import build_mf
from core.fuzzy_arith import fuzzy_add, fuzzy_mul, fuzzy_sub, fuzzy_div, fuzzy_max, fuzzy_min
from core.defuzz import centroid


def test_add_centroids():
    """centroid(a + b) ≈ centroid(a) + centroid(b) для симетричних ФП."""
    a = build_mf({"mf_type": "trimf", "params": [1.0, 2.0, 3.0]}).discretize()
    b = build_mf({"mf_type": "trimf", "params": [4.0, 5.0, 6.0]}).discretize()
    s = fuzzy_add(a, b)
    assert abs(centroid(s) - (centroid(a) + centroid(b))) < 0.05


def test_mul_centroids_symmetric():
    a = build_mf({"mf_type": "trimf", "params": [0.9, 1.0, 1.1]}).discretize()
    b = build_mf({"mf_type": "trimf", "params": [1.9, 2.0, 2.1]}).discretize()
    p = fuzzy_mul(a, b)
    assert abs(centroid(p) - 2.0) < 0.05


def test_add_degenerate_equals_classical():
    """Додавання двох вироджених = додавання чисел."""
    a = build_mf({"mf_type": "trimf", "params": [3.0, 3.0, 3.0]}).discretize()
    b = build_mf({"mf_type": "trimf", "params": [4.0, 4.0, 4.0]}).discretize()
    s = fuzzy_add(a, b)
    assert abs(centroid(s) - 7.0) < 0.1


def test_mul_degenerate_equals_classical():
    a = build_mf({"mf_type": "trimf", "params": [2.0, 2.0, 2.0]}).discretize()
    b = build_mf({"mf_type": "trimf", "params": [3.0, 3.0, 3.0]}).discretize()
    p = fuzzy_mul(a, b)
    assert abs(centroid(p) - 6.0) < 0.1


def test_sub_centroids():
    a = build_mf({"mf_type": "trimf", "params": [9.0, 10.0, 11.0]}).discretize()
    b = build_mf({"mf_type": "trimf", "params": [2.0, 3.0, 4.0]}).discretize()
    d = fuzzy_sub(a, b)
    assert abs(centroid(d) - 7.0) < 0.1


def test_div_simple():
    a = build_mf({"mf_type": "trimf", "params": [9.0, 10.0, 11.0]}).discretize()
    b = build_mf({"mf_type": "trimf", "params": [1.9, 2.0, 2.1]}).discretize()
    q = fuzzy_div(a, b)
    assert abs(centroid(q) - 5.0) < 0.2


def test_fuzzy_max_degenerate():
    a = build_mf({"mf_type": "trimf", "params": [3.0, 3.0, 3.0]}).discretize()
    b = build_mf({"mf_type": "trimf", "params": [5.0, 5.0, 5.0]}).discretize()
    assert abs(centroid(fuzzy_max(a, b)) - 5.0) < 0.1


def test_fuzzy_min_degenerate():
    a = build_mf({"mf_type": "trimf", "params": [3.0, 3.0, 3.0]}).discretize()
    b = build_mf({"mf_type": "trimf", "params": [5.0, 5.0, 5.0]}).discretize()
    assert abs(centroid(fuzzy_min(a, b)) - 3.0) < 0.1
