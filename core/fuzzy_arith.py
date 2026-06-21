import numpy as np
from dataclasses import dataclass

DEFAULT_ALPHA_LEVELS = 51


@dataclass
class FuzzySet:
    """Дискретизована функція належності.
    x: відсортований масив значень аргументу.
    mu: масив значень μ ∈ [0,1] тієї ж довжини.
    Інваріант: max(mu) ≈ 1.0 (нормалізована).
    """
    x: np.ndarray
    mu: np.ndarray

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.mu = np.asarray(self.mu, dtype=float)
        assert len(self.x) == len(self.mu), "x and mu must have the same length"
        assert np.all(np.diff(self.x) >= -1e-12), "x must be sorted ascending"
        assert np.all(self.mu >= -1e-9) and np.all(self.mu <= 1.0 + 1e-9)
        self.mu = np.clip(self.mu, 0.0, 1.0)


def to_alpha_cuts(
    fs: FuzzySet,
    n_levels: int = DEFAULT_ALPHA_LEVELS,
) -> list[tuple[float, float, float]]:
    """Перетворює FuzzySet у список (alpha, x_low, x_high).
    Для кожного α-рівня повертає найширший інтервал де mu >= alpha.
    """
    alphas = np.linspace(0, 1, n_levels)
    cuts = []
    for a in alphas:
        idxs = np.where(fs.mu >= a - 1e-12)[0]
        if len(idxs) == 0:
            continue
        cuts.append((float(a), float(fs.x[idxs[0]]), float(fs.x[idxs[-1]])))
    return cuts


def from_alpha_cuts(
    cuts: list[tuple[float, float, float]],
    n_points: int = 401,
) -> FuzzySet:
    """Збирає FuzzySet з набору α-зрізів.
    Для кожної точки рівномірної сітки знаходить максимальний α,
    такий що точка належить α-зрізу.
    """
    if not cuts:
        return FuzzySet(x=np.array([0.0]), mu=np.array([0.0]))
    x_min = min(c[1] for c in cuts)
    x_max = max(c[2] for c in cuts)
    if x_max <= x_min:
        x_max = x_min + 1e-9
    x_grid = np.linspace(x_min, x_max, n_points)
    mu = np.zeros(n_points)
    for alpha, lo, hi in cuts:
        in_cut = (x_grid >= lo - 1e-12) & (x_grid <= hi + 1e-12)
        mu = np.where(in_cut & (mu < alpha), alpha, mu)
    return FuzzySet(x=x_grid, mu=mu)


def _align_cuts(
    a: FuzzySet,
    b: FuzzySet,
    n_levels: int = DEFAULT_ALPHA_LEVELS,
) -> tuple[list, list]:
    """Повертає α-зрізи обох ФП на спільній сітці α-рівнів."""
    cuts_a = to_alpha_cuts(a, n_levels)
    cuts_b = to_alpha_cuts(b, n_levels)
    # Якщо довжини не збігаються — відрізаємо до меншої (крайні α-рівні)
    n = min(len(cuts_a), len(cuts_b))
    return cuts_a[:n], cuts_b[:n]


def fuzzy_add(a: FuzzySet, b: FuzzySet) -> FuzzySet:
    """⊕: [a_lo+b_lo, a_hi+b_hi] для кожного α-рівня."""
    cuts_a, cuts_b = _align_cuts(a, b)
    result = []
    for (alpha, lo_a, hi_a), (_, lo_b, hi_b) in zip(cuts_a, cuts_b):
        result.append((alpha, lo_a + lo_b, hi_a + hi_b))
    return from_alpha_cuts(result)


def fuzzy_sub(a: FuzzySet, b: FuzzySet) -> FuzzySet:
    """⊖: [a_lo-b_hi, a_hi-b_lo] для кожного α-рівня."""
    cuts_a, cuts_b = _align_cuts(a, b)
    result = []
    for (alpha, lo_a, hi_a), (_, lo_b, hi_b) in zip(cuts_a, cuts_b):
        result.append((alpha, lo_a - hi_b, hi_a - lo_b))
    return from_alpha_cuts(result)


def fuzzy_mul(a: FuzzySet, b: FuzzySet) -> FuzzySet:
    """⊗: min/max усіх 4 добутків кутів інтервалів."""
    cuts_a, cuts_b = _align_cuts(a, b)
    result = []
    for (alpha, lo_a, hi_a), (_, lo_b, hi_b) in zip(cuts_a, cuts_b):
        candidates = [lo_a * lo_b, lo_a * hi_b, hi_a * lo_b, hi_a * hi_b]
        result.append((alpha, min(candidates), max(candidates)))
    return from_alpha_cuts(result)


def fuzzy_div(a: FuzzySet, b: FuzzySet, eps: float = 1e-9) -> FuzzySet:
    """⊘: вимагає що 0 ∉ носій b. Через 4 ділення кутів інтервалів."""
    cuts_a, cuts_b = _align_cuts(a, b)
    result = []
    for (alpha, lo_a, hi_a), (_, lo_b, hi_b) in zip(cuts_a, cuts_b):
        # Захист від ділення на нуль
        if abs(lo_b) < eps:
            lo_b = eps
        if abs(hi_b) < eps:
            hi_b = eps
        # Якщо знаменник змінює знак — пропускаємо рівень
        if lo_b * hi_b < 0:
            continue
        candidates = [lo_a / lo_b, lo_a / hi_b, hi_a / lo_b, hi_a / hi_b]
        result.append((alpha, min(candidates), max(candidates)))
    return from_alpha_cuts(result)


def fuzzy_max(a: FuzzySet, b: FuzzySet) -> FuzzySet:
    """⊻: [max(lo_a,lo_b), max(hi_a,hi_b)] для кожного α-рівня."""
    cuts_a, cuts_b = _align_cuts(a, b)
    result = []
    for (alpha, lo_a, hi_a), (_, lo_b, hi_b) in zip(cuts_a, cuts_b):
        result.append((alpha, max(lo_a, lo_b), max(hi_a, hi_b)))
    return from_alpha_cuts(result)


def fuzzy_min(a: FuzzySet, b: FuzzySet) -> FuzzySet:
    """⊼: [min(lo_a,lo_b), min(hi_a,hi_b)] для кожного α-рівня."""
    cuts_a, cuts_b = _align_cuts(a, b)
    result = []
    for (alpha, lo_a, hi_a), (_, lo_b, hi_b) in zip(cuts_a, cuts_b):
        result.append((alpha, min(lo_a, lo_b), min(hi_a, hi_b)))
    return from_alpha_cuts(result)


def fuzzy_scalar_sub_from_one(a: FuzzySet) -> FuzzySet:
    """1 ⊖ a: [1-a_hi, 1-a_lo] для кожного α-рівня."""
    cuts = to_alpha_cuts(a)
    result = []
    for alpha, lo, hi in cuts:
        result.append((alpha, 1.0 - hi, 1.0 - lo))
    return from_alpha_cuts(result)


def fuzzy_mul_scalar(a: FuzzySet, k: float) -> FuzzySet:
    """k · a: множення нечіткого числа на чітку константу k.
    k > 0: [k·lo, k·hi]; k < 0: [k·hi, k·lo]; k = 0: [0, 0].
    """
    cuts = to_alpha_cuts(a)
    result = []
    for alpha, lo, hi in cuts:
        if k >= 0:
            result.append((alpha, k * lo, k * hi))
        else:
            result.append((alpha, k * hi, k * lo))
    return from_alpha_cuts(result)
