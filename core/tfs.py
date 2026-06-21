from .fuzzy_arith import (
    FuzzySet,
    fuzzy_add,
    fuzzy_mul,
    fuzzy_div,
    fuzzy_sub,
    fuzzy_max,
    fuzzy_min,
    fuzzy_scalar_sub_from_one,
    fuzzy_mul_scalar,
)
from .membership import build_mf

TFSResult = dict  # {"P": FuzzySet, "MT": FuzzySet, "DT": FuzzySet}

TFS_REGISTRY: dict[str, callable] = {}


def register_tfs(name: str):
    def decorator(fn):
        TFS_REGISTRY[name] = fn
        return fn
    return decorator


@register_tfs("RR")
def evaluate_rr(operations: list[dict]) -> TFSResult:
    """ТФС-1: послідовне виконання n РО.
    B = ∏Bᵢ, M(T) = ∑M(Tᵢ), D(T) = ∑D(Tᵢ).
    """
    assert len(operations) >= 1
    P  = build_mf(operations[0]["params"]["B1"]).discretize()
    MT = build_mf(operations[0]["params"]["MT"]).discretize()
    DT = build_mf(operations[0]["params"]["DT"]).discretize()
    for op in operations[1:]:
        P  = fuzzy_mul(P,  build_mf(op["params"]["B1"]).discretize())
        MT = fuzzy_add(MT, build_mf(op["params"]["MT"]).discretize())
        DT = fuzzy_add(DT, build_mf(op["params"]["DT"]).discretize())
    return {"P": P, "MT": MT, "DT": DT}


@register_tfs("RK")
def evaluate_rk(operations: list[dict]) -> TFSResult:
    """ТФС-2: РО з контролем без обмеження циклів.
    P = B1*K11 / (1 - (B1*K10 + B0*K00))
    MT = (MTr + MTk) / denom
    """
    assert len(operations) == 2
    r, k = operations
    assert r["type"] == "R" and k["type"] == "K"

    B1  = build_mf(r["params"]["B1"]).discretize()
    MTr = build_mf(r["params"]["MT"]).discretize()
    DTr = build_mf(r["params"]["DT"]).discretize()
    K11 = build_mf(k["params"]["K11"]).discretize()
    K00 = build_mf(k["params"]["K00"]).discretize()
    MTk = build_mf(k["params"]["MT"]).discretize()
    DTk = build_mf(k["params"]["DT"]).discretize()

    B0  = fuzzy_scalar_sub_from_one(B1)
    K10 = fuzzy_scalar_sub_from_one(K11)

    # denom = 1 - (B1*K10 + B0*K00)
    denom_inner = fuzzy_add(fuzzy_mul(B1, K10), fuzzy_mul(B0, K00))
    denom = fuzzy_scalar_sub_from_one(denom_inner)

    P  = fuzzy_div(fuzzy_mul(B1, K11), denom)
    MT = fuzzy_div(fuzzy_add(MTr, MTk), denom)
    DT = fuzzy_div(fuzzy_add(DTr, DTk), denom)

    return {"P": P, "MT": MT, "DT": DT}


@register_tfs("RKR1")
def evaluate_rkr1(operations: list[dict]) -> TFSResult:
    """ТФС-3: РО з контролем і одним доопрацюванням.
    P = B1*K11 + (B0*K00 + B1*K10)*B21
    MT = MTr + MTk + (B0*K00 + B1*K10)*MTr2
    """
    assert len(operations) == 3
    r, k, r2 = operations
    assert r["type"] == "R" and k["type"] == "K" and r2["type"] == "R2"

    B1   = build_mf(r["params"]["B1"]).discretize()
    MTr  = build_mf(r["params"]["MT"]).discretize()
    DTr  = build_mf(r["params"]["DT"]).discretize()
    K11  = build_mf(k["params"]["K11"]).discretize()
    K00  = build_mf(k["params"]["K00"]).discretize()
    MTk  = build_mf(k["params"]["MT"]).discretize()
    DTk  = build_mf(k["params"]["DT"]).discretize()
    B21  = build_mf(r2["params"]["B1"]).discretize()
    MTr2 = build_mf(r2["params"]["MT"]).discretize()
    DTr2 = build_mf(r2["params"]["DT"]).discretize()

    B0  = fuzzy_scalar_sub_from_one(B1)
    K10 = fuzzy_scalar_sub_from_one(K11)

    # Ймовірність повторного циклу: q = B0*K00 + B1*K10
    q = fuzzy_add(fuzzy_mul(B0, K00), fuzzy_mul(B1, K10))

    # P = B1*K11 + q*B21
    P = fuzzy_add(fuzzy_mul(B1, K11), fuzzy_mul(q, B21))

    # MT = MTr + MTk + q*MTr2
    MT = fuzzy_add(fuzzy_add(MTr, MTk), fuzzy_mul(q, MTr2))
    DT = fuzzy_add(fuzzy_add(DTr, DTk), fuzzy_mul(q, DTr2))

    return {"P": P, "MT": MT, "DT": DT}


@register_tfs("RKR")
def evaluate_rkr(operations: list[dict]) -> TFSResult:
    """ТФС-4: циклічна РКР з нескінченною геометричною прогресією.
    operations = [R, K, R2]. Контроль 2-го циклу — ті самі параметри K.

    P = (B1*K11 + q*B21*K11) / (1 - q*(1 - B21*K11))
    M(T) = (MTr + MTk + q*(MTr2 + MTk)) / denom
    де q = B0*K00 + B1*K10.
    """
    assert len(operations) == 3
    r, k, r2 = operations

    B1   = build_mf(r["params"]["B1"]).discretize()
    MTr  = build_mf(r["params"]["MT"]).discretize()
    DTr  = build_mf(r["params"]["DT"]).discretize()
    K11  = build_mf(k["params"]["K11"]).discretize()
    K00  = build_mf(k["params"]["K00"]).discretize()
    MTk  = build_mf(k["params"]["MT"]).discretize()
    DTk  = build_mf(k["params"]["DT"]).discretize()
    B21  = build_mf(r2["params"]["B1"]).discretize()
    MTr2 = build_mf(r2["params"]["MT"]).discretize()
    DTr2 = build_mf(r2["params"]["DT"]).discretize()

    B0  = fuzzy_scalar_sub_from_one(B1)
    K10 = fuzzy_scalar_sub_from_one(K11)

    # q = ймовірність потрапити в повторний цикл
    q = fuzzy_add(fuzzy_mul(B0, K00), fuzzy_mul(B1, K10))

    # success_2nd = B21 * K11
    success_2nd = fuzzy_mul(B21, K11)
    # denom = 1 - q * (1 - success_2nd)
    denom = fuzzy_scalar_sub_from_one(
        fuzzy_mul(q, fuzzy_scalar_sub_from_one(success_2nd))
    )

    # P = (B1*K11 + q*success_2nd) / denom
    P = fuzzy_div(
        fuzzy_add(fuzzy_mul(B1, K11), fuzzy_mul(q, success_2nd)),
        denom
    )

    # MT = (MTr + MTk + q*(MTr2 + MTk)) / denom
    MT = fuzzy_div(
        fuzzy_add(fuzzy_add(MTr, MTk), fuzzy_mul(q, fuzzy_add(MTr2, MTk))),
        denom
    )
    DT = fuzzy_div(
        fuzzy_add(fuzzy_add(DTr, DTk), fuzzy_mul(q, fuzzy_add(DTr2, DTk))),
        denom
    )

    return {"P": P, "MT": MT, "DT": DT}


@register_tfs("PAR")
def evaluate_par(operations: list[dict]) -> TFSResult:
    """ТФС-5: паралельне виконання n РО.
    operations[0] = {"composer": "AND" | "OR" | "XOR"}.
    operations[1:] = РО {type, params:{B1, MT, DT}}.

    AND: P = ∏Pᵢ, MT = max(MTᵢ).
    OR:  P = 1 - ∏(1-Pᵢ), MT = min(MTᵢ).
    XOR: P як AND, MT як min (спрощення MVP).
    """
    assert len(operations) >= 2
    composer = operations[0].get("composer", "AND")
    ops = operations[1:]

    Ps  = [build_mf(op["params"]["B1"]).discretize() for op in ops]
    MTs = [build_mf(op["params"]["MT"]).discretize() for op in ops]
    DTs = [build_mf(op["params"]["DT"]).discretize() for op in ops]

    if composer == "AND":
        P = Ps[0]
        for p in Ps[1:]:
            P = fuzzy_mul(P, p)
        MT = MTs[0]
        for m in MTs[1:]:
            MT = fuzzy_max(MT, m)
        DT = DTs[0]
        for d in DTs[1:]:
            DT = fuzzy_max(DT, d)

    elif composer == "OR":
        Q = fuzzy_scalar_sub_from_one(Ps[0])
        for p in Ps[1:]:
            Q = fuzzy_mul(Q, fuzzy_scalar_sub_from_one(p))
        P = fuzzy_scalar_sub_from_one(Q)
        MT = MTs[0]
        for m in MTs[1:]:
            MT = fuzzy_min(MT, m)
        DT = DTs[0]
        for d in DTs[1:]:
            DT = fuzzy_min(DT, d)

    elif composer == "XOR":
        # XOR: рівно одна альтернатива з n виконується (рівні ваги 1/n).
        # P_XOR  = Σ w·Bᵢ           (зважене середнє ймовірностей)
        # MT_XOR = Σ w·MTᵢ          (зважене середнє часів)
        # DT_XOR = Σ w²·DTᵢ         (властивість дисперсії суми)
        n = len(ops)
        w = 1.0 / n
        P = fuzzy_mul_scalar(Ps[0], w)
        for p in Ps[1:]:
            P = fuzzy_add(P, fuzzy_mul_scalar(p, w))
        MT = fuzzy_mul_scalar(MTs[0], w)
        for m in MTs[1:]:
            MT = fuzzy_add(MT, fuzzy_mul_scalar(m, w))
        DT = fuzzy_mul_scalar(DTs[0], w * w)
        for d in DTs[1:]:
            DT = fuzzy_add(DT, fuzzy_mul_scalar(d, w * w))

    else:
        raise ValueError(f"Unknown composer: {composer}. Use AND/OR/XOR.")

    return {"P": P, "MT": MT, "DT": DT}
