"""Юніт-тести для validate_payload (backend-валідація вхідних даних)."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from app import validate_payload


def _op(b1_params, mt_params, dt_params, mf='trimf'):
    return {"type": "R", "params": {
        "B1": {"mf_type": mf, "params": b1_params},
        "MT": {"mf_type": "trimf", "params": mt_params},
        "DT": {"mf_type": "trimf", "params": dt_params},
    }}


def test_validate_rejects_probability_above_one():
    payload = {"tfs_type": "RR", "operations": [
        _op([0.5, 0.9, 1.5], [1, 2, 3], [0.1, 0.2, 0.3])  # 1.5 > 1
    ]}
    errors = validate_payload(payload)
    assert any("ймовірність" in e for e in errors)


def test_validate_rejects_negative_time():
    payload = {"tfs_type": "RR", "operations": [
        _op([0.1, 0.5, 0.9], [-1, 0, 1], [0.1, 0.2, 0.3])  # -1 < 0
    ]}
    errors = validate_payload(payload)
    assert any("від'ємним" in e for e in errors)


def test_validate_rejects_unsorted_trimf():
    payload = {"tfs_type": "RR", "operations": [
        _op([0.5, 0.3, 0.9], [1, 2, 3], [0.1, 0.2, 0.3])  # 0.3 < 0.5
    ]}
    errors = validate_payload(payload)
    assert any("a ≤ b ≤ c" in e for e in errors)


def test_validate_accepts_valid_payload():
    payload = {"tfs_type": "RR", "operations": [
        {"type": "R", "params": {
            "B1": {"mf_type": "trimf",  "params": [0.95, 0.97, 0.99]},
            "MT": {"mf_type": "gaussmf","params": [1.0, 5.0]},
            "DT": {"mf_type": "trimf",  "params": [0.1, 0.2, 0.3]},
        }}
    ]}
    assert validate_payload(payload) == []


def test_validate_rejects_negative_directive_time():
    payload = {"tfs_type": "RR", "directive_time": -5,
               "operations": [_op([0.9, 0.95, 1.0], [1, 2, 3], [0.1, 0.2, 0.3])]}
    errors = validate_payload(payload)
    assert any("Директивний час" in e for e in errors)


def test_validate_rejects_too_many_operations():
    ops = [_op([0.9, 0.95, 1.0], [1, 2, 3], [0.1, 0.2, 0.3])] * 21
    errors = validate_payload({"tfs_type": "RR", "operations": ops})
    assert any("максимум" in e for e in errors)