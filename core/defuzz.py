import numpy as np
from .fuzzy_arith import FuzzySet


def centroid(fs: FuzzySet) -> float:
    """Центроїдна дефазифікація: ∑(xᵢ·μᵢ) / ∑μᵢ."""
    s = fs.mu.sum()
    if s < 1e-12:
        return float(fs.x[len(fs.x) // 2])
    return float(np.sum(fs.x * fs.mu) / s)


def support(fs: FuzzySet, eps: float = 1e-6) -> tuple[float, float]:
    """Носій: інтервал де μ > eps."""
    mask = fs.mu > eps
    if not np.any(mask):
        return (float(fs.x[0]), float(fs.x[-1]))
    idxs = np.where(mask)[0]
    return (float(fs.x[idxs[0]]), float(fs.x[idxs[-1]]))


def core(fs: FuzzySet, eps: float = 1e-6) -> tuple[float, float]:
    """Ядро: інтервал де μ ≥ 1 - eps."""
    mask = fs.mu >= 1.0 - eps
    if not np.any(mask):
        i = int(np.argmax(fs.mu))
        return (float(fs.x[i]), float(fs.x[i]))
    idxs = np.where(mask)[0]
    return (float(fs.x[idxs[0]]), float(fs.x[idxs[-1]]))
