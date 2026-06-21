import json
from pathlib import Path
from core.tfs import TFS_REGISTRY
from core.defuzz import centroid


def test_basic_verification_example():
    """Базовий верифікаційний приклад: 7 послідовних РО з трикутними ФП.

    Еталонні значення (класична детермінована теорія):
      B ≈ 0.9760, M(T) ≈ 39.6 с, D(T) ≈ 3.6 с²
    Це прямі формули ТФС-1: B = ∏Bᵢ, M(T) = ∑M(Tᵢ), D(T) = ∑D(Tᵢ).
    Нечіткий результат (центроїди) має збігатись з еталоном у межах допуску.
    """
    example_path = Path(__file__).parent.parent / "examples" / "default_example.json"
    with open(example_path) as f:
        data = json.load(f)
    result = TFS_REGISTRY["RR"](data["operations"])
    B = centroid(result["P"])
    M = centroid(result["MT"])
    D = centroid(result["DT"])
    assert abs(B - 0.9760) < 0.01
    assert abs(M - 39.6) < 0.5
    assert abs(D - 3.6) < 0.1