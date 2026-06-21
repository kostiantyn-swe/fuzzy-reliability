from abc import ABC, abstractmethod
import numpy as np
from .fuzzy_arith import FuzzySet

DEFAULT_GRID_POINTS = 401


class MembershipFunction(ABC):
    """Базовий клас усіх функцій належності."""

    @abstractmethod
    def value(self, x: np.ndarray) -> np.ndarray:
        """Векторно обчислює μ(x)."""

    @abstractmethod
    def support(self) -> tuple[float, float]:
        """Інтервал [x_min, x_max] де μ > 0."""

    def discretize(
        self,
        n_points: int = DEFAULT_GRID_POINTS,
        x_range: tuple[float, float] | None = None,
    ) -> FuzzySet:
        """Дискретизація на n_points точок у межах x_range або support()."""
        if x_range is None:
            x_range = self._effective_support()
        x = np.linspace(x_range[0], x_range[1], n_points)
        return FuzzySet(x=x, mu=self.value(x))

    def _effective_support(self) -> tuple[float, float]:
        return self.support()


MEMBERSHIP_REGISTRY: dict[str, type["MembershipFunction"]] = {}


def register_membership(name: str):
    """Декоратор для реєстрації нового типу ФП."""
    def decorator(cls):
        MEMBERSHIP_REGISTRY[name] = cls
        return cls
    return decorator


def build_mf(spec: dict) -> MembershipFunction:
    """Фабрика: будує об'єкт ФП за словником {mf_type, params}."""
    mf_type = spec["mf_type"]
    params = spec["params"]
    if mf_type not in MEMBERSHIP_REGISTRY:
        raise ValueError(
            f"Unknown mf_type: {mf_type}. "
            f"Available: {list(MEMBERSHIP_REGISTRY.keys())}"
        )
    return MEMBERSHIP_REGISTRY[mf_type](*params)


@register_membership("trimf")
class TriangularMF(MembershipFunction):
    """Трикутна ФП. Параметри: [a, b, c], a ≤ b ≤ c."""

    def __init__(self, a: float, b: float, c: float):
        assert a <= b <= c, f"trimf params: a≤b≤c required, got {a},{b},{c}"
        self.a, self.b, self.c = float(a), float(b), float(c)

    def value(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mu = np.zeros_like(x)
        # Вироджений випадок a==b==c — дельта-функція
        if self.a == self.b == self.c:
            return np.where(np.isclose(x, self.b), 1.0, 0.0)
        if self.b > self.a:
            mask = (x > self.a) & (x <= self.b)
            mu[mask] = (x[mask] - self.a) / (self.b - self.a)
        if self.c > self.b:
            mask = (x > self.b) & (x < self.c)
            mu[mask] = (self.c - x[mask]) / (self.c - self.b)
        # У вершині b завжди μ=1
        mu = np.where(np.isclose(x, self.b), 1.0, mu)
        return mu

    def support(self) -> tuple[float, float]:
        if self.a == self.c:
            eps = max(abs(self.b) * 1e-6, 1e-9)
            return (self.b - eps, self.b + eps)
        return (self.a, self.c)


@register_membership("trapmf")
class TrapezoidalMF(MembershipFunction):
    """Трапецієдальна ФП. Параметри: [a, b, c, d], a ≤ b ≤ c ≤ d."""

    def __init__(self, a: float, b: float, c: float, d: float):
        assert a <= b <= c <= d, f"trapmf params: a≤b≤c≤d required"
        self.a, self.b, self.c, self.d = map(float, (a, b, c, d))

    def value(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mu = np.zeros_like(x)
        if self.b > self.a:
            mask = (x > self.a) & (x < self.b)
            mu[mask] = (x[mask] - self.a) / (self.b - self.a)
        mask = (x >= self.b) & (x <= self.c)
        mu[mask] = 1.0
        if self.d > self.c:
            mask = (x > self.c) & (x < self.d)
            mu[mask] = (self.d - x[mask]) / (self.d - self.c)
        return mu

    def support(self) -> tuple[float, float]:
        if self.a == self.d:
            eps = max(abs(self.a) * 1e-6, 1e-9)
            return (self.a - eps, self.d + eps)
        return (self.a, self.d)


@register_membership("gaussmf")
class GaussianMF(MembershipFunction):
    """Гаусівська ФП. Параметри: [sigma, c]. μ(x) = exp(-0.5*((x-c)/sigma)²)"""

    def __init__(self, sigma: float, c: float):
        assert sigma > 0, f"gaussmf: sigma must be > 0, got {sigma}"
        self.sigma = float(sigma)
        self.c = float(c)

    def value(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.exp(-0.5 * ((x - self.c) / self.sigma) ** 2)

    def support(self) -> tuple[float, float]:
        return (self.c - 4 * self.sigma, self.c + 4 * self.sigma)

    def _effective_support(self) -> tuple[float, float]:
        return self.support()


# ── Допоміжна функція для сигмоїдних ФП ──────────────────────────────────────

def _sigmf_val(x: np.ndarray, a: float, c: float) -> np.ndarray:
    """σ(x) = 1 / (1 + exp(-a*(x-c))). Векторизована."""
    return 1.0 / (1.0 + np.exp(-a * (x - c)))


# ── Нові 10 типів ФП ──────────────────────────────────────────────────────────

@register_membership("gauss2mf")
class Gauss2MF(MembershipFunction):
    """Двостороння гаусівська ФП. Параметри: [σ₁, c₁, σ₂, c₂], c₁ ≤ c₂.
    μ(x) = exp(-½·((x-c₁)/σ₁)²) при x≤c₁; 1 при c₁<x<c₂; exp(-½·((x-c₂)/σ₂)²) при x≥c₂.
    """

    def __init__(self, sigma1: float, c1: float, sigma2: float, c2: float):
        assert sigma1 > 0 and sigma2 > 0
        assert c1 <= c2, "gauss2mf: c1 must be <= c2"
        self.sigma1, self.c1 = float(sigma1), float(c1)
        self.sigma2, self.c2 = float(sigma2), float(c2)

    def value(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mu = np.ones_like(x)
        left = x <= self.c1
        right = x >= self.c2
        mu[left] = np.exp(-0.5 * ((x[left] - self.c1) / self.sigma1) ** 2)
        mu[right] = np.exp(-0.5 * ((x[right] - self.c2) / self.sigma2) ** 2)
        return mu

    def support(self) -> tuple[float, float]:
        return (self.c1 - 4 * self.sigma1, self.c2 + 4 * self.sigma2)


@register_membership("gbellmf")
class GeneralizedBellMF(MembershipFunction):
    """Узагальнена дзвоноподібна. Параметри: [a, b, c], a>0, b>0.
    μ(x) = 1 / (1 + |((x-c)/a)|^(2b)).
    """

    def __init__(self, a: float, b: float, c: float):
        assert a > 0 and b > 0
        self.a, self.b, self.c = float(a), float(b), float(c)

    def value(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return 1.0 / (1.0 + np.abs((x - self.c) / self.a) ** (2.0 * self.b))

    def support(self) -> tuple[float, float]:
        return (self.c - 4.0 * self.a, self.c + 4.0 * self.a)


@register_membership("sigmf")
class SigmoidMF(MembershipFunction):
    """Сигмоїдна ФП. Параметри: [a, c], a≠0.
    μ(x) = 1 / (1 + exp(-a*(x-c))).
    """

    def __init__(self, a: float, c: float):
        assert a != 0, "sigmf: a must be non-zero"
        self.a, self.c = float(a), float(c)

    def value(self, x: np.ndarray) -> np.ndarray:
        return _sigmf_val(np.asarray(x, dtype=float), self.a, self.c)

    def support(self) -> tuple[float, float]:
        half = 5.0 / abs(self.a)
        return (self.c - half, self.c + half)


@register_membership("dsigmf")
class DifferenceSigmoidMF(MembershipFunction):
    """Різниця двох сигмоїд. Параметри: [a₁, c₁, a₂, c₂].
    μ(x) = clip(sigmf(a₁,c₁) − sigmf(a₂,c₂), 0, 1).
    """

    def __init__(self, a1: float, c1: float, a2: float, c2: float):
        self.a1, self.c1 = float(a1), float(c1)
        self.a2, self.c2 = float(a2), float(c2)

    def value(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.clip(_sigmf_val(x, self.a1, self.c1) - _sigmf_val(x, self.a2, self.c2),
                       0.0, 1.0)

    def support(self) -> tuple[float, float]:
        w1 = 5.0 / max(abs(self.a1), 1e-9)
        w2 = 5.0 / max(abs(self.a2), 1e-9)
        return (min(self.c1, self.c2) - max(w1, w2),
                max(self.c1, self.c2) + max(w1, w2))


@register_membership("psigmf")
class ProductSigmoidMF(MembershipFunction):
    """Добуток двох сигмоїд. Параметри: [a₁, c₁, a₂, c₂].
    μ(x) = sigmf(a₁,c₁) · sigmf(a₂,c₂).
    """

    def __init__(self, a1: float, c1: float, a2: float, c2: float):
        self.a1, self.c1 = float(a1), float(c1)
        self.a2, self.c2 = float(a2), float(c2)

    def value(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return _sigmf_val(x, self.a1, self.c1) * _sigmf_val(x, self.a2, self.c2)

    def support(self) -> tuple[float, float]:
        w = 5.0 / max(abs(self.a1), abs(self.a2), 1e-9)
        return (min(self.c1, self.c2) - w, max(self.c1, self.c2) + w)


@register_membership("zmf")
class ZMF(MembershipFunction):
    """Z-подібна (спадна). Параметри: [a, b], a<b.
    μ(x) = 1 при x≤a; 1−2·((x-a)/(b-a))² при a<x≤mid; 2·((x-b)/(b-a))² при mid<x<b; 0 при x≥b.
    """

    def __init__(self, a: float, b: float):
        assert a < b, "zmf: a must be < b"
        self.a, self.b = float(a), float(b)

    def value(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mu = np.zeros_like(x)
        mid = (self.a + self.b) / 2.0
        span = self.b - self.a
        mu[x <= self.a] = 1.0
        m1 = (x > self.a) & (x <= mid)
        mu[m1] = 1.0 - 2.0 * ((x[m1] - self.a) / span) ** 2
        m2 = (x > mid) & (x < self.b)
        mu[m2] = 2.0 * ((x[m2] - self.b) / span) ** 2
        return mu

    def support(self) -> tuple[float, float]:
        return (self.a, self.b)


@register_membership("smf")
class SMF(MembershipFunction):
    """S-подібна (зростаюча). Параметри: [a, b], a<b.
    μ(x) = 0 при x≤a; 2·((x-a)/(b-a))² при a<x≤mid; 1−2·((x-b)/(b-a))² при mid<x<b; 1 при x≥b.
    """

    def __init__(self, a: float, b: float):
        assert a < b, "smf: a must be < b"
        self.a, self.b = float(a), float(b)

    def value(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        mu = np.zeros_like(x)
        mid = (self.a + self.b) / 2.0
        span = self.b - self.a
        mu[x >= self.b] = 1.0
        m1 = (x >= self.a) & (x <= mid)
        mu[m1] = 2.0 * ((x[m1] - self.a) / span) ** 2
        m2 = (x > mid) & (x < self.b)
        mu[m2] = 1.0 - 2.0 * ((x[m2] - self.b) / span) ** 2
        return mu

    def support(self) -> tuple[float, float]:
        return (self.a, self.b)


@register_membership("pimf")
class PiMF(MembershipFunction):
    """П-подібна ФП. Параметри: [a, b, c, d], a≤b≤c≤d.
    = smf(a,b) · zmf(c,d).
    """

    def __init__(self, a: float, b: float, c: float, d: float):
        assert a <= b <= c <= d
        self.a, self.b, self.c, self.d = map(float, (a, b, c, d))

    def value(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        # SMF частина на [a, b]
        smf = np.zeros_like(x)
        if self.b > self.a:
            mid_s = (self.a + self.b) / 2.0
            span_s = self.b - self.a
            smf[x >= self.b] = 1.0
            m1 = (x >= self.a) & (x <= mid_s)
            smf[m1] = 2.0 * ((x[m1] - self.a) / span_s) ** 2
            m2 = (x > mid_s) & (x < self.b)
            smf[m2] = 1.0 - 2.0 * ((x[m2] - self.b) / span_s) ** 2
        else:
            smf[x >= self.b] = 1.0
        # ZMF частина на [c, d]
        zmf = np.zeros_like(x)
        if self.d > self.c:
            mid_z = (self.c + self.d) / 2.0
            span_z = self.d - self.c
            zmf[x <= self.c] = 1.0
            m3 = (x > self.c) & (x <= mid_z)
            zmf[m3] = 1.0 - 2.0 * ((x[m3] - self.c) / span_z) ** 2
            m4 = (x > mid_z) & (x < self.d)
            zmf[m4] = 2.0 * ((x[m4] - self.d) / span_z) ** 2
        else:
            zmf[x <= self.c] = 1.0
        return smf * zmf

    def support(self) -> tuple[float, float]:
        return (self.a, self.d)


@register_membership("linsmf")
class LinearSMF(MembershipFunction):
    """Лінійна зростаюча ФП. Параметри: [a, b], a<b.
    μ(x) = 0 при x≤a; (x-a)/(b-a) при a<x<b; 1 при x≥b.
    """

    def __init__(self, a: float, b: float):
        assert a < b, "linsmf: a must be < b"
        self.a, self.b = float(a), float(b)

    def value(self, x: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(x, dtype=float) - self.a) / (self.b - self.a), 0.0, 1.0)

    def support(self) -> tuple[float, float]:
        return (self.a, self.b)


@register_membership("linzmf")
class LinearZMF(MembershipFunction):
    """Лінійна спадна ФП. Параметри: [a, b], a<b.
    μ(x) = 1 при x≤a; 1-(x-a)/(b-a) при a<x<b; 0 при x≥b.
    """

    def __init__(self, a: float, b: float):
        assert a < b, "linzmf: a must be < b"
        self.a, self.b = float(a), float(b)

    def value(self, x: np.ndarray) -> np.ndarray:
        return np.clip(1.0 - (np.asarray(x, dtype=float) - self.a) / (self.b - self.a),
                       0.0, 1.0)

    def support(self) -> tuple[float, float]:
        return (self.a, self.b)
