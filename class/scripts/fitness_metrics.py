"""
GA fitness metrics: chi2, AIC, BIC, or DIC (lower is better).

For Gaussian / Cobaya-style fits::

    AIC = chi2 + 2 k
    BIC = chi2 + k * ln(N)
    DIC = 2 Dbar - Dhat   (from posterior / training logL; see posterior_diagnostics)

``k`` = number of free Cobaya parameters with a prior (cosmology + active
``w`` coeffs). Unused ``w0`` / ``wa`` fixed to 0 are not counted.

``N`` = effective number of data points (override with ``GA_BIC_N_DATA``).
Default 1600 ≈ Pantheon+SH0ES + DESI BAO order of magnitude.

Paper default metric is ``dic`` (env ``GA_FITNESS_METRIC``). Pair with the
unconstrained-DE gate in ``posterior_diagnostics`` so flat posteriors do not
win on raw fit quality alone.
"""
from __future__ import annotations

import math
import os
import re
from typing import Tuple


def fitness_metric() -> str:
    """Return ``chi2`` | ``aic`` | ``bic`` | ``dic`` (default ``dic``)."""
    m = os.environ.get("GA_FITNESS_METRIC", "dic").strip().lower()
    if m not in ("chi2", "aic", "bic", "dic"):
        raise ValueError(
            f"GA_FITNESS_METRIC must be chi2|aic|bic|dic, got {m!r}"
        )
    return m


def n_data() -> float:
    """Effective data-point count for BIC (env ``GA_BIC_N_DATA``)."""
    raw = os.environ.get("GA_BIC_N_DATA", "1600").strip()
    n = float(raw)
    if not math.isfinite(n) or n <= 1.0:
        raise ValueError(f"GA_BIC_N_DATA must be > 1, got {raw!r}")
    return n


def count_free_params(expr: str) -> int:
    """
    Count free continuous parameters for this ``w(a)`` expression.

    Uses ``params.build_ga_params`` so unused ``w0``/``wa`` (fixed to 0)
    are excluded; always-sampled cosmology priors are included.
    """
    from params import build_ga_params

    params = build_ga_params(expr or "", template_params=None)
    return sum(
        1
        for spec in params.values()
        if isinstance(spec, dict) and "prior" in spec
    )


def count_de_params(expr: str) -> int:
    """Number of distinct DE coefficient symbols in the expression."""
    found = set()
    for raw in re.findall(r"\b(w\d+|wa|w0)\b", expr or ""):
        found.add({"w0": "w0", "wa": "wa"}.get(raw, raw))
    return len(found)


def score_from_chi2(chi2: float, expr: str) -> Tuple[float, float, int]:
    """
    Convert χ² → GA fitness for chi2 / AIC / BIC.

    For DIC use ``score_from_dic`` instead (needs posterior logL).

    Returns ``(fitness, chi2, k)``. Failed / non-finite χ² → ``(+inf, chi2, k)``.
    """
    k = count_free_params(expr)
    if chi2 is None or not math.isfinite(float(chi2)):
        return float("inf"), float("inf"), k

    chi2_f = float(chi2)
    metric = fitness_metric()
    if metric == "bic":
        fitness = chi2_f + k * math.log(n_data())
    elif metric == "aic":
        fitness = chi2_f + 2.0 * k
    elif metric == "dic":
        # Caller should prefer score_from_dic; fall back to BIC-like if no DIC.
        fitness = chi2_f + k * math.log(n_data())
    else:
        fitness = chi2_f
    return fitness, chi2_f, k


def score_from_dic(dic: float, chi2: float, expr: str) -> Tuple[float, float, int]:
    """Return ``(DIC_fitness, chi2, k)``; non-finite DIC → ``+inf``."""
    k = count_free_params(expr)
    chi2_f = float(chi2) if chi2 is not None and math.isfinite(float(chi2)) else float("inf")
    if dic is None or not math.isfinite(float(dic)):
        return float("inf"), chi2_f, k
    return float(dic), chi2_f, k


def metric_label() -> str:
    m = fitness_metric()
    if m == "bic":
        return f"BIC (chi2 + k*ln(N), N={n_data():.0f})"
    if m == "aic":
        return "AIC (chi2 + 2k)"
    if m == "dic":
        return "DIC (2*Dbar - Dhat from CLiENT logL)"
    return "chi2"
