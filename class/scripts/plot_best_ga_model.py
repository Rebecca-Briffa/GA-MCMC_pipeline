#!/usr/bin/env python3
"""
Standalone GetDist plotter for a finished CLiENT run (any gen_*_model_*).

No GA image / Cobaya / CLASS required — only::

    pip install numpy h5py pyyaml matplotlib getdist

Usage::

    python plot_best_ga_model.py /path/to/gen_X_model_Y
    python plot_best_ga_model.py /path/to/GA_run_dir   # reads GA_summary, picks best

For a new long CLiENT + plot, use ``post_ga.py`` instead.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
from getdist import MCSamples, plots
from matplotlib import pyplot as plt


def _best_model_id_from_summary(summary: Path) -> str:
    # Prefer fitness (col 3); require status success.
    # Formats: ID chi2 fitness [k] status backend [mask] expr
    line_re_k = re.compile(
        r"^(gen_\d+_model_\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\S+\s+(.*)$"
    )
    line_re_old = re.compile(
        r"^(gen_\d+_model_\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\S+\s+(.*)$"
    )
    best_id, best_fit = None, float("inf")
    for line in summary.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        m = line_re_k.match(line)
        if m and m.group(5).lower() in ("success", "failed"):
            model_id, _chi2_s, fit_s, _k, status, _rest = m.groups()
        else:
            m = line_re_old.match(line)
            if not m:
                continue
            model_id, _chi2_s, fit_s, status, _rest = m.groups()
        if status.lower() != "success":
            continue
        try:
            fit = float(fit_s)
        except ValueError:
            continue
        if np.isfinite(fit) and fit < best_fit:
            best_fit = fit
            best_id = model_id
    if not best_id:
        raise SystemExit(f"No successful models in {summary}")
    print(f"Best model from {summary.name}: {best_id}  fitness={best_fit:.4f}")
    return best_id


def _find_model_folder(ga_root: Path, model_id: str) -> Path | None:
    for cand in (ga_root / model_id, ga_root / model_id / model_id):
        if cand.is_dir() and (cand / "class").is_dir():
            return cand
    if ga_root.is_dir():
        for child in ga_root.iterdir():
            if not child.is_dir():
                continue
            if child.name == model_id and (child / "class").is_dir():
                return child
            nested = child / model_id
            if nested.is_dir() and (nested / "class").is_dir():
                return nested
    return None


def _resolve_model_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "class").is_dir():
        return path
    if path.name == "class" and path.parent.is_dir():
        return path.parent
    nested = path / path.name
    if nested.is_dir() and (nested / "class").is_dir():
        return nested
    summary = path if path.name == "GA_summary.txt" else path / "GA_summary.txt"
    if summary.is_file():
        root = summary.parent
        mid = _best_model_id_from_summary(summary)
        found = _find_model_folder(root, mid)
        if found is None:
            raise SystemExit(
                f"Best model is {mid}, but no folder with class/ found under {root}"
            )
        return found
    raise SystemExit(
        f"Expected a model folder containing class/, or a GA dir with "
        f"GA_summary.txt, got:\n  {path}"
    )

def _yaml_has_priors(path: Path) -> bool:
    try:
        info = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace")) or {}
    except Exception:
        return False
    params = info.get("params") or {}
    return any(isinstance(spec, dict) and "prior" in spec for spec in params.values())


def _find_yaml(model_root: Path, explicit: Path | None, run_dir: Path | None = None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Missing yaml: {p}")
        return p
    candidates: list[Path] = []
    # Prefer Cobaya yaml from the run we are plotting
    if run_dir is not None and run_dir.is_dir():
        candidates.extend(sorted(run_dir.glob("*_client.yaml")))
        candidates.extend(sorted(run_dir.glob("*.yaml")))
    cfg = model_root / "class" / "scripts" / "client_config"
    candidates.append(cfg / f"{model_root.name}.yaml")
    if cfg.is_dir():
        candidates.extend(sorted(cfg.glob("*.yaml")))
    # Fall back to newest run folders
    runs = model_root / "class" / "scripts" / "client_runs"
    if runs.is_dir():
        for run in sorted(runs.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if run_dir is not None and run.resolve() == run_dir.resolve():
                continue
            candidates.extend(sorted(run.glob("*_client.yaml")))
            candidates.extend(sorted(run.glob("*.yaml")))
    seen: set[Path] = set()
    for p in candidates:
        rp = p.resolve() if p.exists() else p
        if rp in seen or not p.is_file():
            continue
        seen.add(rp)
        if _yaml_has_priors(p):
            return p
    raise SystemExit(
        f"No Cobaya/CLiENT yaml with params.*.prior found under {model_root}"
    )

def sampled_names(yaml_path: Path) -> list[str]:
    info = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="replace"))
    return [
        name
        for name, spec in (info.get("params") or {}).items()
        if isinstance(spec, dict) and "prior" in spec
    ]


def latest_run(model_root: Path) -> Path:
    base = model_root / "class" / "scripts" / "client_runs"
    if not base.is_dir():
        raise SystemExit(f"No client_runs folder at {base}")
    runs = [p for p in base.iterdir() if p.is_dir()]
    if not runs:
        raise SystemExit(f"No CLiENT runs in {base}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def latest_chain(run_dir: Path) -> Path:
    files = sorted((run_dir / "training_chains").glob("chain_it_*.h5"))
    if not files:
        raise SystemExit(f"No chain_it_*.h5 in {run_dir / 'training_chains'}")
    return files[-1]


def training_xy(run_dir: Path):
    xs, ys = [], []
    tdir = run_dir / "training_data"
    if not tdir.is_dir():
        raise SystemExit(f"No training_data/ in {run_dir}")
    for path in sorted(tdir.glob("data_it_*.h5")):
        with h5py.File(path, "r") as f:
            xs.append(f["data"]["x"][...])
            ys.append(f["data"]["y"][...])
    if not xs:
        raise SystemExit(f"No data_it_*.h5 in {tdir}")
    return np.vstack(xs), np.concatenate(ys)


def w_in_formula(model_root: Path) -> list[str]:
    """w's that appear in this model's w(a), from user_eq.h if present."""
    header = model_root / "class" / "include" / "user_eq.h"
    if not header.is_file():
        return []
    formula = ""
    for line in header.read_text(encoding="utf-8", errors="replace").splitlines():
        if "#define" in line and "FLUID_W_FLD" in line and ") (" in line:
            formula = "(" + line.split(") (", 1)[1]
            break
    found: list[str] = []
    for raw in re.findall(r"\b(w\d+|wa|w0)\b", formula):
        name = {"w0": "w0_fld", "wa": "wa_fld"}.get(raw, raw)
        if name not in found:
            found.append(name)
    return found


def read_r1(run_dir: Path) -> tuple[float | None, bool | None]:
    """Parse last R-1 line from metrics.log if present."""
    metrics = run_dir / "metrics.log"
    if not metrics.is_file():
        return None, None
    last_r1, last_ok = None, None
    in_block = False
    for line in metrics.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Gelman-Rubin" in line or "R-1" in line and "converged" in line.lower():
            in_block = True
            continue
        if in_block:
            if line.strip().startswith("---") or not line.strip():
                if last_r1 is not None:
                    break
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[0] not in ("it", "avg", "tot"):
                try:
                    last_r1 = float(parts[1])
                    last_ok = parts[2].lower().startswith("t")
                except ValueError:
                    pass
    return last_r1, last_ok


def load_flat_chain(
    chain_path: Path,
    names: list[str],
    *,
    burn_frac: float,
    thin: int,
) -> np.ndarray:
    with h5py.File(chain_path, "r") as f:
        chain = np.asarray(f["mcmc"]["chain"][...])  # steps, walkers, params
    if chain.ndim != 3:
        raise SystemExit(f"Unexpected chain shape {chain.shape} in {chain_path}")
    if chain.shape[-1] != len(names):
        print(
            f"WARNING: chain has {chain.shape[-1]} cols, yaml has {len(names)} names — "
            "using min length",
            file=sys.stderr,
        )
        n = min(chain.shape[-1], len(names))
        chain = chain[..., :n]
        names[:] = names[:n]
    nsteps = chain.shape[0]
    burn = int(max(0, min(nsteps - 1, round(burn_frac * nsteps))))
    chain = chain[burn:: max(1, thin)]
    return chain.reshape(-1, chain.shape[-1])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Standalone GetDist plots from a CLiENT client_runs/ folder"
    )
    p.add_argument(
        "model",
        type=Path,
        help="Model folder (gen_*_model_*) or GA run dir with GA_summary.txt",
    )
    p.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Specific client_runs/<stamp>_... dir (default: newest)",
    )
    p.add_argument("--yaml", type=Path, default=None, help="Param yaml (default: auto)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <model>/best_model_plots)",
    )
    p.add_argument(
        "--params",
        type=str,
        default=None,
        help="Comma-separated GetDist params (default: H0,omegas + active w's)",
    )
    p.add_argument(
        "--add-w1",
        action="store_true",
        help="Add derived w(1)=w0_fld+w4 when both exist (helps break degeneracy view)",
    )
    p.add_argument(
        "--burn-frac",
        type=float,
        default=0.3,
        help="Fraction of chain steps to discard as burn-in (default: 0.3)",
    )
    p.add_argument(
        "--thin",
        type=int,
        default=1,
        help="Extra thinning after burn-in (default: 1 = keep all)",
    )
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Triangle plot title (default: model name + run stamp)",
    )
    args = p.parse_args(argv)

    model_root = _resolve_model_root(args.model)
    run_dir = (
        args.run.expanduser().resolve()
        if args.run is not None
        else latest_run(model_root)
    )
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")
    yaml_path = _find_yaml(model_root, args.yaml, run_dir=run_dir)
    out = (args.out or (model_root / "best_model_plots")).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    names = sampled_names(yaml_path)
    if not names:
        raise SystemExit(f"No sampled params (prior:) in {yaml_path}")

    x, y = training_xy(run_dir)
    if x.shape[1] != len(names):
        n = min(x.shape[1], len(names))
        print(
            f"WARNING: training x has {x.shape[1]} cols vs {len(names)} names; "
            f"using first {n}",
            file=sys.stderr,
        )
        names = names[:n]
        x = x[:, :n]
    ibest = int(np.argmax(y))
    best = {n: float(x[ibest, i]) for i, n in enumerate(names)}
    chi2 = float(-2.0 * y[ibest])
    active_w = [n for n in w_in_formula(model_root) if n in best]
    r1, r1_ok = read_r1(run_dir)

    print(f"model_dir : {model_root}")
    print(f"run_dir   : {run_dir}")
    print(f"yaml      : {yaml_path}")
    print(f"out       : {out}")
    print(f"chi2 (max training logL → -2y) : {chi2:.4f}")
    if r1 is not None:
        print(f"R-1 (metrics.log) : {r1:.6g}  converged={r1_ok}")
    print("\nAll sampled parameters (best training point)")
    for n, v in best.items():
        print(f"  {n:<16} {v:.8g}")
    if active_w:
        print("\nw in this w(a)")
        for n in active_w:
            print(f"  {n:<16} {best[n]:.8g}")

    txt = out / f"{model_root.name}_params.txt"
    with txt.open("w", encoding="utf-8") as f:
        f.write(f"model: {model_root.name}\n")
        f.write(f"chi2_training_approx: {chi2:.6f}\n")
        f.write(f"run: {run_dir}\n")
        if r1 is not None:
            f.write(f"R-1: {r1:.8g}\n")
            f.write(f"R-1_converged: {r1_ok}\n")
        f.write("\n=== all sampled parameters ===\n")
        for n, v in best.items():
            f.write(f"{n:<16} {v:.8g}\n")
        f.write("\n=== w coefficients in this w(a) ===\n")
        for n in active_w:
            f.write(f"{n:<16} {best[n]:.8g}\n")

    fig, axes = plt.subplots(2, 1, figsize=(8, 0.4 * (len(best) + len(active_w) + 8)))
    for ax in axes:
        ax.axis("off")
    axes[0].set_title("All sampled parameters")
    axes[0].table(
        cellText=[[n, f"{v:.6g}"] for n, v in best.items()],
        colLabels=["parameter", "value"],
        loc="center",
        cellLoc="left",
    )
    w_cells = [[n, f"{best[n]:.6g}"] for n in active_w] or [["(none)", "—"]]
    axes[1].set_title("w coefficients in this w(a)")
    axes[1].table(cellText=w_cells, colLabels=["parameter", "value"], loc="center", cellLoc="left")
    fig.tight_layout()
    table_pdf = out / f"{model_root.name}_params_table.pdf"
    fig.savefig(table_pdf, bbox_inches="tight")
    plt.close(fig)

    chain_path = latest_chain(run_dir)
    names_plot = list(names)
    flat = load_flat_chain(
        chain_path, names_plot, burn_frac=args.burn_frac, thin=args.thin
    )

    labels = list(names_plot)
    if args.add_w1 and "w0_fld" in names_plot and "w4" in names_plot:
        i0 = names_plot.index("w0_fld")
        i4 = names_plot.index("w4")
        flat = np.column_stack([flat, flat[:, i0] + flat[:, i4]])
        names_plot.append("w_a1")
        labels.append(r"w(1)=w_0+w_4")

    samples = MCSamples(
        samples=flat,
        names=names_plot,
        labels=labels,
        label="CLiENT",
    )

    if args.params:
        triangle_params = [s.strip() for s in args.params.split(",") if s.strip()]
        if args.add_w1 and "w_a1" in names_plot and "w_a1" not in triangle_params:
            triangle_params.append("w_a1")
    else:
        triangle_params = [
            n for n in ("H0", "omega_cdm", "omega_b", "Omega_m") if n in names_plot
        ]
        triangle_params += [n for n in active_w if n in names_plot]
        if args.add_w1 and "w_a1" in names_plot:
            triangle_params.append("w_a1")
    missing = [n for n in triangle_params if n not in names_plot]
    if missing:
        raise SystemExit(f"Unknown --params (not in chain/yaml): {missing}")
    if not triangle_params:
        triangle_params = names_plot

    title = args.title or f"{model_root.name}  ({run_dir.name})"
    g = plots.get_subplot_plotter(width_inch=max(8, 1.1 * len(triangle_params)))
    g.triangle_plot(samples, triangle_params, filled=True, title_limit=1)
    g.fig.suptitle(title, y=1.02)
    triangle_pdf = out / f"{model_root.name}_getdist_triangle.pdf"
    g.fig.savefig(triangle_pdf, bbox_inches="tight")
    plt.close(g.fig)

    print(f"\nGetDist from {chain_path.name}  (burn_frac={args.burn_frac}, thin={args.thin})")
    print(f"  n_flat samples used: {flat.shape[0]}")
    print(f"Wrote {txt}")
    print(f"Wrote {table_pdf}")
    print(f"Wrote {triangle_pdf}")


if __name__ == "__main__":
    main()
