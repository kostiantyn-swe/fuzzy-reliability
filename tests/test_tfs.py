import pytest
from core.tfs import TFS_REGISTRY
from core.defuzz import centroid


def trimf(value):
    """Вироджена трикутна для перевірки на класичних формулах."""
    return {"mf_type": "trimf", "params": [value, value, value]}


def test_rr_sequential_classical():
    """ТФС-1 на 7 РО — верифікаційний приклад.
    Очікуване: B ≈ 0.9760, M(T) = 39.6, D(T) = 3.6
    """
    Bs = [0.999, 0.998, 0.997, 0.995, 0.989, 0.999, 0.999]
    Ms = [5.1, 3.0, 8.5, 1.7, 2.3, 10.0, 9.0]
    Ds = [0.3, 0.4, 0.7, 0.1, 0.2, 1.1, 0.8]
    operations = [
        {"type": "R", "params": {"B1": trimf(b), "MT": trimf(m), "DT": trimf(d)}}
        for b, m, d in zip(Bs, Ms, Ds)
    ]
    result = TFS_REGISTRY["RR"](operations)
    expected_B = 1.0
    for b in Bs:
        expected_B *= b
    assert abs(centroid(result["P"]) - expected_B) < 0.01 * expected_B
    assert abs(centroid(result["MT"]) - sum(Ms)) < 0.01 * sum(Ms)
    assert abs(centroid(result["DT"]) - sum(Ds)) < 0.01 * sum(Ds)


def test_rk_classical():
    """ТФС-2 на вироджених ФП — порівняння з класичною формулою."""
    B1, K11, K00 = 0.998, 0.99, 0.975
    MTr, MTk = 5.0, 4.0
    DTr, DTk = 0.4, 0.7
    operations = [
        {"type": "R", "params": {"B1": trimf(B1), "MT": trimf(MTr), "DT": trimf(DTr)}},
        {"type": "K", "params": {"K11": trimf(K11), "K00": trimf(K00),
                                  "MT": trimf(MTk), "DT": trimf(DTk)}},
    ]
    result = TFS_REGISTRY["RK"](operations)
    B0 = 1 - B1
    K10 = 1 - K11
    denom = 1 - (B1 * K10 + B0 * K00)
    expected_B = B1 * K11 / denom
    expected_MT = (MTr + MTk) / denom
    assert abs(centroid(result["P"]) - expected_B) < 0.02 * expected_B
    assert abs(centroid(result["MT"]) - expected_MT) < 0.02 * expected_MT


def test_rkr1_classical():
    B1, K11, K00, B21 = 0.99, 0.99, 0.97, 0.999
    MTr, MTk, MTr2 = 5.0, 4.0, 3.0
    DTr, DTk, DTr2 = 0.3, 0.5, 0.2
    operations = [
        {"type": "R",  "params": {"B1": trimf(B1), "MT": trimf(MTr),  "DT": trimf(DTr)}},
        {"type": "K",  "params": {"K11": trimf(K11), "K00": trimf(K00),
                                   "MT": trimf(MTk), "DT": trimf(DTk)}},
        {"type": "R2", "params": {"B1": trimf(B21), "MT": trimf(MTr2), "DT": trimf(DTr2)}},
    ]
    result = TFS_REGISTRY["RKR1"](operations)
    B0 = 1 - B1
    K10 = 1 - K11
    expected_B = B1 * K11 + (B0 * K00 + B1 * K10) * B21
    assert abs(centroid(result["P"]) - expected_B) < 0.02


def test_rkr_classical():
    """ТФС-4: циклічна РКР — перевірка проти класичної формули."""
    B1, K11, K00, B21 = 0.99, 0.99, 0.97, 0.999
    MTr, MTk, MTr2 = 5.0, 4.0, 3.0
    DTr, DTk, DTr2 = 0.3, 0.5, 0.2
    operations = [
        {"type": "R",  "params": {"B1": trimf(B1), "MT": trimf(MTr),  "DT": trimf(DTr)}},
        {"type": "K",  "params": {"K11": trimf(K11), "K00": trimf(K00),
                                   "MT": trimf(MTk), "DT": trimf(DTk)}},
        {"type": "R2", "params": {"B1": trimf(B21), "MT": trimf(MTr2), "DT": trimf(DTr2)}},
    ]
    result = TFS_REGISTRY["RKR"](operations)
    B0, K10 = 1 - B1, 1 - K11
    q = B0 * K00 + B1 * K10
    succ2 = B21 * K11
    denom = 1 - q * (1 - succ2)
    expected_B = (B1 * K11 + q * succ2) / denom
    assert abs(centroid(result["P"]) - expected_B) < 0.02


def test_par_and_classical():
    """ТФС-5 AND: P = P1*P2, MT = max(MT1, MT2)."""
    operations = [
        {"composer": "AND"},
        {"type": "R", "params": {"B1": trimf(0.99), "MT": trimf(5.0), "DT": trimf(0.3)}},
        {"type": "R", "params": {"B1": trimf(0.98), "MT": trimf(7.0), "DT": trimf(0.4)}},
    ]
    result = TFS_REGISTRY["PAR"](operations)
    assert abs(centroid(result["P"]) - 0.99 * 0.98) < 0.02
    assert abs(centroid(result["MT"]) - 7.0) < 0.2


def test_par_or_classical():
    """ТФС-5 OR: P = 1-(1-P1)*(1-P2), MT = min(MT1, MT2)."""
    operations = [
        {"composer": "OR"},
        {"type": "R", "params": {"B1": trimf(0.9),  "MT": trimf(5.0), "DT": trimf(0.3)}},
        {"type": "R", "params": {"B1": trimf(0.85), "MT": trimf(7.0), "DT": trimf(0.4)}},
    ]
    result = TFS_REGISTRY["PAR"](operations)
    expected_B = 1 - (1 - 0.9) * (1 - 0.85)
    assert abs(centroid(result["P"]) - expected_B) < 0.02
    assert abs(centroid(result["MT"]) - 5.0) < 0.2


def test_par_xor():
    """ТФС-5 XOR: рівно одна альтернатива з n (рівні ваги 1/n).
    P_XOR  = Σ(1/n · Bᵢ),  очікується (0.92+0.90+0.91)/3 ≈ 0.910
    MT_XOR = Σ(1/n · MTᵢ), очікується (5+4+5)/3 ≈ 4.667
    Результат відрізняється від AND (≈0.752) та OR (≈0.999).
    """
    Bs  = [0.92, 0.90, 0.91]
    MTs = [5.0,  4.0,  5.0]
    DTs = [0.3,  0.2,  0.25]
    operations = [
        {"composer": "XOR"},
        *[
            {"type": "R", "params": {
                "B1": trimf(b), "MT": trimf(m), "DT": trimf(d),
            }}
            for b, m, d in zip(Bs, MTs, DTs)
        ],
    ]
    result = TFS_REGISTRY["PAR"](operations)
    c_P  = centroid(result["P"])
    c_MT = centroid(result["MT"])

    # XOR: зважене середнє
    assert 0.905 <= c_P  <= 0.915, f"P centroid={c_P:.4f}, expected ~0.910"
    assert 4.65  <= c_MT <= 4.68,  f"MT centroid={c_MT:.4f}, expected ~4.667"

    # Переконуємось, що результат відрізняється від AND і OR
    result_and = TFS_REGISTRY["PAR"]([{"composer": "AND"}] + operations[1:])
    result_or  = TFS_REGISTRY["PAR"]([{"composer": "OR"}]  + operations[1:])
    assert abs(c_P - centroid(result_and["P"])) > 0.1, "XOR == AND (помилка)"
    assert abs(c_P - centroid(result_or["P"]))  > 0.05, "XOR == OR (помилка)"
