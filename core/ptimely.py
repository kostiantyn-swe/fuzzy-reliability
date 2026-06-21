"""Розрахунок ймовірності своєчасного виконання P_св(T_дир).

Підтримує три методи згідно прикладів наукового керівника:
- "analytical" — аналітичні формули для трикутної/трапецієдальної ФП часу;
- "possibility" — можливісна оцінка: P_св(T_дир) = max{μ_T(t) : t ≤ T_дир};
- "integral"   — інтегральна (нормований cumtrapz, CDF).

Кожен метод повертає PtimelyCurve — пара (t_dir_grid, p_sv_values).
"""
import numpy as np
from dataclasses import dataclass
from .fuzzy_arith import FuzzySet

DEFAULT_T_DIR_POINTS = 500


@dataclass
class PtimelyCurve:
    """Крива P_св(T_дир): для кожного T_дир — ймовірність своєчасності."""
    t_dir:  np.ndarray   # сітка директивного часу
    p_sv:   np.ndarray   # значення P_св ∈ [0, 1]
    method: str          # "analytical", "possibility", "integral"

    def at(self, t_directive: float) -> float:
        """Інтерпольоване значення P_св у заданій точці T_дир."""
        return float(np.interp(t_directive, self.t_dir, self.p_sv))


# ── Метод 1: аналітичний (trimf / trapmf) ────────────────────────────────────

def ptimely_analytical_trimf(a: float, b: float, c: float,
                              t_dir_grid: np.ndarray) -> np.ndarray:
    """Аналітичний P_св для трикутної ФП часу [a, b, c].

    Формула за зразком прикладу 1 керівника:
      0                      якщо T_дир ≤ a
      (T_дир − a) / (b − a)  якщо a < T_дир ≤ b
      1                      якщо T_дир > b

    Спадна гілка праворуч від моди b ігнорується — при T_дир ≥ b
    вважається що виконання гарантоване.
    """
    p = np.zeros_like(t_dir_grid, dtype=float)
    if b > a:
        mask = (t_dir_grid > a) & (t_dir_grid <= b)
        p[mask] = (t_dir_grid[mask] - a) / (b - a)
    p[t_dir_grid > b] = 1.0
    return p


def ptimely_analytical_trapmf(a: float, b: float, c: float, d: float,
                               t_dir_grid: np.ndarray) -> np.ndarray:
    """Аналітичний P_св для трапецієдальної ФП часу [a, b, c, d].

    Формула за зразком прикладу 2 керівника — аналогічно trimf,
    спадна гілка (c, d) ігнорується:
      0                      якщо T_дир ≤ a
      (T_дир − a) / (b − a)  якщо a < T_дир ≤ b
      1                      якщо T_дир > b
    """
    p = np.zeros_like(t_dir_grid, dtype=float)
    if b > a:
        mask = (t_dir_grid > a) & (t_dir_grid <= b)
        p[mask] = (t_dir_grid[mask] - a) / (b - a)
    p[t_dir_grid > b] = 1.0
    return p


# ── Метод 2: можливісний (max μ_T(t) для t ≤ T_дир) ────────────────────────

def ptimely_possibility(MT: FuzzySet, t_dir_grid: np.ndarray) -> np.ndarray:
    """Можливісна оцінка: max μ_T(t) для t ≤ T_дир.

    Використовує кумулятивний максимум по щільній сітці MT.x,
    потім лінійно інтерполює на t_dir_grid. Це дає гладку S-криву
    без сходинкових артефактів (реалізація прикладів 3, 4 керівника).
    """
    cum_max = np.maximum.accumulate(MT.mu)
    return np.interp(t_dir_grid, MT.x, cum_max,
                     left=0.0, right=float(cum_max[-1]))


# ── Метод 3: інтегральний (нормований cumtrapz) ──────────────────────────────

def ptimely_integral(MT: FuzzySet, t_dir_grid: np.ndarray) -> np.ndarray:
    """Інтегральна оцінка: нормований кумулятивний інтеграл ФП часу.

    Реалізація прикладу 5 керівника:
      P_sv_cum = cumtrapz(xT, mf_T_res); P_sv_cum = P_sv_cum / max(P_sv_cum)

    Трактуємо нормовану μ_T як щільність ймовірності і рахуємо CDF.
    """
    cum = np.zeros_like(MT.x)
    if len(MT.x) > 1:
        cum[1:] = np.cumsum(
            0.5 * (MT.mu[:-1] + MT.mu[1:]) * np.diff(MT.x)
        )
    total = cum[-1] if cum[-1] > 1e-12 else 1.0
    cum_norm = cum / total
    return np.interp(t_dir_grid, MT.x, cum_norm, left=0.0, right=1.0)


# ── Універсальний фасад ───────────────────────────────────────────────────────

def compute_ptimely(MT: FuzzySet,
                    t_dir_min: float = 0.0,
                    t_dir_max: float | None = None,
                    n_points: int | None = None,
                    method: str = "auto",
                    mt_spec: dict | None = None) -> PtimelyCurve:
    """Обчислює P_св(T_дир) для заданої ФП часу.

    Параметри:
      MT        — дискретизована ФП часу (FuzzySet).
      t_dir_min — ліва межа сітки T_дир (за замовч. 0).
      t_dir_max — права межа. None → 1.1 × правий кінець носія MT.
      n_points  — кількість точок. None → автопідбір: ~10 точок/с,
                  мін DEFAULT_T_DIR_POINTS, макс 3000.
      method    — "analytical" | "possibility" | "integral" | "auto".
      mt_spec   — специфікація ФП для методу analytical та auto-вибору.

    Повертає: PtimelyCurve.
    """
    if t_dir_max is None:
        t_dir_max = float(MT.x[-1]) * 1.1
    t_dir_min = max(0.0, t_dir_min)
    if n_points is None:
        width = max(t_dir_max - t_dir_min, 1e-9)
        n_points = int(np.clip(width * 10, DEFAULT_T_DIR_POINTS, 3000))
    t_dir_grid = np.linspace(t_dir_min, t_dir_max, n_points)

    if method == "auto":
        method = select_default_method(mt_spec)

    if method == "analytical":
        if mt_spec is None:
            raise ValueError(
                "Method 'analytical' requires mt_spec with mf_type and params"
            )
        mf_type = mt_spec["mf_type"]
        params  = mt_spec["params"]
        if mf_type == "trimf":
            p_sv = ptimely_analytical_trimf(*params, t_dir_grid)
        elif mf_type == "trapmf":
            p_sv = ptimely_analytical_trapmf(*params, t_dir_grid)
        else:
            raise ValueError(
                f"Method 'analytical' supports only trimf/trapmf, got {mf_type}. "
                "Use 'possibility' or 'integral' instead."
            )
    elif method == "possibility":
        p_sv = ptimely_possibility(MT, t_dir_grid)
    elif method == "integral":
        p_sv = ptimely_integral(MT, t_dir_grid)
    else:
        raise ValueError(f"Unknown method: {method!r}. "
                         "Use 'analytical', 'possibility', 'integral', or 'auto'.")

    return PtimelyCurve(t_dir=t_dir_grid, p_sv=p_sv, method=method)


def select_default_method(mt_spec: dict | None) -> str:
    """Обирає метод за замовчуванням згідно прикладів керівника:
      trimf, trapmf → analytical;
      gaussmf та всі інші → possibility.

    Integral не використовується автоматично — потребує явного вибору
    (зазвичай для циклічних структур ТФС-2 за прикладом 5 керівника).
    """
    if mt_spec is None:
        return "possibility"
    if mt_spec.get("mf_type", "") in ("trimf", "trapmf"):
        return "analytical"
    return "possibility"


# ── Вилучення параметрів з результуючої ФП ───────────────────────────────────

def _count_humps(mu: np.ndarray, threshold: float) -> int:
    """Кількість зв'язних областей mu > threshold."""
    count, in_hump = 0, False
    for v in mu > threshold:
        if v and not in_hump:
            count += 1
            in_hump = True
        elif not v:
            in_hump = False
    return count


def extract_trimf_params_from_fuzzyset(
        fs: FuzzySet, eps: float = 0.01) -> tuple[float, float, float]:
    """Вилучає параметри (a, b, c) приблизно трикутної ФП.

    a = лівий край носія  (перший x де μ > eps·max)
    b = пік              (x в точці argmax μ)
    c = правий край носія (останній x де μ > eps·max)

    Піднімає ValueError, якщо ФП має більше одного значущого горба
    (більше одного зв'язного підйому вище 50% від максимуму).
    """
    mu, x = fs.mu, fs.x
    max_mu = float(mu.max())
    if max_mu < 1e-9:
        raise ValueError("Result MT has no significant values")
    if _count_humps(mu, 0.5 * max_mu) > 1:
        raise ValueError("Result MT is not approximately triangular")

    mask = mu > eps * max_mu
    idxs = np.where(mask)[0]
    return float(x[idxs[0]]), float(x[int(np.argmax(mu))]), float(x[idxs[-1]])


def extract_trapmf_params_from_fuzzyset(
        fs: FuzzySet, eps: float = 0.01) -> tuple[float, float, float, float]:
    """Вилучає параметри (a, b, c, d) приблизно трапецієдальної ФП.

    a = лівий край носія
    b = ліва межа плато  (перший x де μ ≥ (1-eps)·max)
    c = права межа плато (останній x де μ ≥ (1-eps)·max)
    d = правий край носія

    Піднімає ValueError, якщо плато вироджене (< 3 точок) або ФП не унімодальна.
    """
    mu, x = fs.mu, fs.x
    max_mu = float(mu.max())
    if max_mu < 1e-9:
        raise ValueError("Result MT has no significant values")
    if _count_humps(mu, 0.5 * max_mu) > 1:
        raise ValueError("Result MT is not approximately trapezoidal")

    plateau_mask = mu >= (1.0 - eps) * max_mu
    plateau_idxs = np.where(plateau_mask)[0]

    support_idxs = np.where(mu > eps * max_mu)[0]
    support_width = float(x[support_idxs[-1]] - x[support_idxs[0]])
    plateau_width = float(x[plateau_idxs[-1]] - x[plateau_idxs[0]])

    # Плато повинно займати щонайменше 5 % ширини носія,
    # інакше це вироджений «пік» трикутної ФП — використовуй extract_trimf.
    if support_width < 1e-9 or plateau_width / support_width < 0.05:
        raise ValueError(
            "Result MT plateau is too narrow — use extract_trimf instead"
        )

    return (float(x[support_idxs[0]]),
            float(x[plateau_idxs[0]]),
            float(x[plateau_idxs[-1]]),
            float(x[support_idxs[-1]]))