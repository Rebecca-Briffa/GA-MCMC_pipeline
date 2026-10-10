"""
DIC and posterior-constraint checks from a CLiENT run folder.

Used by the GA after each model evaluation when ``GA_FITNESS_METRIC=dic``
and/or ``GA_REQUIRE_CONSTRAINED=1``.

DIC (Spiegelhalter)::

    D_i = -2 log L_i
    Dbar = mean(D)
    Dhat = -2 log L_max   (best training point)
    p_D  = Dbar - Dhat
    DIC  = Dbar + p_D = 2 Dbar - Dhat

Unconstrained DE coefficient (default rule)::

    A free DE coeff is unconstrained if its 95% equal-tailed interval
    covers at least ``GA_UNCONSTRAINED_FRAC`` (default 0.85) of the prior
    width. This catches flat posteriors like ``w6`` in a GetDist triangle.
    Failed models get ``fail_reason=unconstrained:<names>`` in ``GA_summary``.
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def require_constrained() -> bool:
    raw = os.environ.get("GA_REQUIRE_CONSTRAINED", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def unconstrained_frac() -> float:
    raw = os.environ.get("GA_UNCONSTRAINED_FRAC", "0.85").strip()
    frac = float(raw)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"GA_UNCONSTRAINED_FRAC must be in (0,1], got {raw!r}")
    return frac


def burn_frac() -> float:
    raw = os.environ.get("GA_CONSTRAINT_BURN_FRAC", "0.3").strip()
    return float(raw)


def de_cobaya_names(expr: str) -> list[str]:
    """Cobaya/CLiENT names for DE coeffs appearing in ``expr``."""
    found: list[str] = []
    for raw in re.findall(r"\b(w\d+|wa|w0)\b", expr or ""):
        name = {"w0": "w0_fld", "wa": "wa_fld"}.get(raw, raw)
        if name not in found:
            found.append(name)
    return found


def dic_from_logl(y: np.ndarray) -> tuple[float, float, float]:
    """
    Return ``(DIC, Dbar, p_D)`` from an array of log-likelihood values.
    """
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if y.size == 0:
        return float("inf"), float("inf"), float("inf")
    D = -2.0 * y
    Dbar = float(np.mean(D))
    Dhat = float(-2.0 * np.max(y))
    p_D = Dbar - Dhat
    return Dbar + p_D, Dbar, p_D


def _load_training_logl(run_dir: Path) -> np.ndarray | None:
    import h5py

    tdir = run_dir / "training_data"
    if not tdir.is_dir():
        return None
    ys = []
    for path in sorted(tdir.glob("data_it_*.h5")):
        with h5py.File(path, "r") as f:
            ys.append(np.asarray(f["data"]["y"][...], dtype=float))
    if not ys:
        return None
    return np.concatenate(ys)


def _latest_chain(run_dir: Path) -> Path | None:
    files = sorted((run_dir / "training_chains").glob("chain_it_*.h5"))
    return files[-1] if files else None


def _sampled_priors(yaml_path: Path) -> dict[str, tuple[float, float]]:
    info = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="replace")) or {}
    out: dict[str, tuple[float, float]] = {}
    for name, spec in (info.get("params") or {}).items():
        if not isinstance(spec, dict) or "prior" not in spec:
            continue
        prior = spec["prior"] or {}
        if "min" in prior and "max" in prior:
            out[name] = (float(prior["min"]), float(prior["max"]))
    return out


def _find_yaml(run_dir: Path, model_root: Path | None = None) -> Path | None:
    cands: list[Path] = []
    cands.extend(sorted(run_dir.glob("*_client.yaml")))
    cands.extend(sorted(run_dir.glob("*.yaml")))
    if model_root is not None:
        cfg = model_root / "class" / "scripts" / "client_config"
        if cfg.is_dir():
            cands.extend(sorted(cfg.glob("*.yaml")))
    for p in cands:
        if p.is_file() and "prior" in p.read_text(encoding="utf-8", errors="replace"):
            return p
    return None


def _flat_chain(chain_path: Path, n_params: int, burn: float) -> np.ndarray:
    import h5py

    with h5py.File(chain_path, "r") as f:
        chain = np.asarray(f["mcmc"]["chain"][...], dtype=float)
    if chain.ndim != 3:
        raise ValueError(f"unexpected chain shape {chain.shape}")
    n = min(n_params, chain.shape[-1])
    chain = chain[..., :n]
    nsteps = chain.shape[0]
    i0 = int(max(0, min(nsteps - 1, round(burn * nsteps))))
    return chain[i0:].reshape(-1, n)


def unconstrained_names(
    flat: np.ndarray,
    names: list[str],
    priors: dict[str, tuple[float, float]],
    check_names: list[str],
    *,
    frac: float | None = None,
) -> list[str]:
    """Return DE param names whose 95% interval spans too much of the prior."""
    frac = unconstrained_frac() if frac is None else frac
    bad: list[str] = []
    for name in check_names:
        if name not in names or name not in priors:
            continue
        j = names.index(name)
        lo_p, hi_p = priors[name]
        width = hi_p - lo_p
        if width <= 0:
            continue
        col = flat[:, j]
        col = col[np.isfinite(col)]
        if col.size < 20:
            bad.append(name)
            continue
        q_lo, q_hi = np.percentile(col, [2.5, 97.5])
        if (q_hi - q_lo) / width >= frac:
            bad.append(name)
    return bad


def diagnose_client_run(
    run_dir: str | Path | None,
    expr: str,
    *,
    model_root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compute DIC and unconstrained DE coeffs from a CLiENT ``client_runs`` folder.

    Returns a dict with keys::
        dic, Dbar, p_D, unconstrained, fail_reason, ok
    """
    out: dict[str, Any] = {
        "dic": float("inf"),
        "Dbar": float("inf"),
        "p_D": float("inf"),
        "unconstrained": [],
        "fail_reason": "",
        "ok": False,
    }
    if run_dir is None:
        out["fail_reason"] = "no_posterior"
        return out

    run_path = Path(run_dir)
    if not run_path.is_dir():
        out["fail_reason"] = "no_posterior"
        return out

    y = _load_training_logl(run_path)
    if y is None:
        out["fail_reason"] = "no_training_logl"
        return out

    dic, Dbar, p_D = dic_from_logl(y)
    out["dic"] = dic
    out["Dbar"] = Dbar
    out["p_D"] = p_D

    if not require_constrained():
        out["ok"] = math.isfinite(dic)
        return out

    root = Path(model_root) if model_root is not None else None
    yaml_path = _find_yaml(run_path, root)
    chain_path = _latest_chain(run_path)
    if yaml_path is None or chain_path is None:
        out["fail_reason"] = "no_chain_for_constraint"
        return out

    priors = _sampled_priors(yaml_path)
    names = list(priors.keys())
    if not names:
        out["fail_reason"] = "no_priors"
        return out

    try:
        flat = _flat_chain(chain_path, len(names), burn_frac())
    except Exception as exc:
        out["fail_reason"] = f"chain_read_error:{type(exc).__name__}"
        return out

    # Align name list length to chain columns
    names = names[: flat.shape[1]]
    de_names = [n for n in de_cobaya_names(expr) if n in names]
    bad = unconstrained_names(flat, names, priors, de_names)
    out["unconstrained"] = bad
    if bad:
        out["fail_reason"] = "unconstrained:" + ",".join(bad)
        out["ok"] = False
        return out

    out["ok"] = math.isfinite(dic)
    if not out["ok"]:
        out["fail_reason"] = "nonfinite_dic"
    return out


def newest_client_run(model_class_dir: str | Path) -> Path | None:
    base = Path(model_class_dir) / "scripts" / "client_runs"
    if not base.is_dir():
        return None
    runs = [p for p in base.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)
