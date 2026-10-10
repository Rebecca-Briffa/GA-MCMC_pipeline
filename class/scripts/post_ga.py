#!/usr/bin/env python3
"""
Post-GA analysis for the winning model (standalone — not part of the GA loop).

1. Optional: rebuild CLASS for the winning ``w(a)``
2. Run CLiENT with profile ``ga_final`` / ``ga_final_hpc`` (T=1, longer emcee)
3. Write best-fit table + GetDist triangle from the new run

The winning model is **not hard-coded**. Pass either:
  - a specific ``gen_*_model_*`` folder, or
  - a GA run directory / ``GA_summary.txt`` (auto-picks lowest χ²), or
  - nothing (uses ``./GA_summary.txt``).

Unused ``w0_fld`` / ``wa_fld`` are fixed to 0 when absent from the expression
(see ``params.build_ga_params``).

Standalone::

    export CLIENT_ROOT=/path/to/client_public
    conda activate clienv

    # Auto-pick best model from GA_summary.txt in this directory:
    python post_ga.py
    python post_ga.py /path/to/GA_run_dir --profile ga_final_hpc

    # Or point at one model folder explicitly:
    python post_ga.py /path/to/gen_X_model_Y

Plots: ``<model>/post_ga_plots/``. Re-plot only: ``plot_best_ga_model.py <model>``.

After the GA (``GA_RUN_POST=1``, default), ``GA_MCMC_new.py`` calls this on the
winner from ``GA_summary.txt``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("COBAYA_NOMPI", "1")

_MODEL_ID_RE = re.compile(r"^gen_\d+_model_\d+$")
# ---------------------------------------------------------------------------


def _resolve_model_dirs(model_path: Path) -> tuple[str, Path, Path]:
    model_path = model_path.expanduser().resolve()
    if (model_path / "class").is_dir():
        model_root = model_path
    elif model_path.name == "class" and model_path.is_dir():
        model_root = model_path.parent
    else:
        # Nested copy: .../name/name/class
        nested = model_path / model_path.name
        if nested.is_dir() and (nested / "class").is_dir():
            model_root = nested
        else:
            raise SystemExit(f"Expected a folder containing class/, got: {model_path}")
    return model_root.name, model_root, model_root / "class"


def best_from_ga_summary(
    summary_path: Path | None = None,
    *,
    search_roots: list[Path] | None = None,
) -> tuple[str, str, float]:
    """
    Return ``(model_id, expr, chi2)`` for the best successful model in
    ``GA_summary.txt``.

    Ranking uses the **fitness** column (BIC/AIC/chi2 — whatever the GA used).
    The returned third value is still the recorded **chi2** (for logging / MAP
    comparison). Supports both old lines (no ``k``) and new lines (with ``k``).
    """
    candidates: list[Path] = []
    if summary_path is not None:
        p = summary_path.expanduser().resolve()
        if p.is_dir():
            p = p / "GA_summary.txt"
        candidates.append(p)
    for root in search_roots or []:
        candidates.append((root / "GA_summary.txt").resolve())
    candidates.append((Path.cwd() / "GA_summary.txt").resolve())

    seen: set[Path] = set()
    path = None
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.is_file():
            path = c
            break
    if path is None:
        raise SystemExit(
            "No GA_summary.txt found. Pass a GA run directory, a model folder, "
            "or --expr / --summary."
        )

    best_id: str | None = None
    best_expr: str | None = None
    best_chi2 = float("inf")
    best_fit = float("inf")
    # Formats (status must be success|failed):
    #   ID chi2 fitness k status backend expr
    #   ID chi2 fitness k status backend mask expr   (mask = 0/1 bitstring)
    # Older: ID chi2 fitness status backend expr
    line_re_new = re.compile(
        r"^(gen_\d+_model_\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\S+\s+(.*)$"
    )
    line_re_old = re.compile(
        r"^(gen_\d+_model_\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\S+\s+(.*)$"
    )
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        m = line_re_new.match(line)
        if m and m.group(5).lower() in ("success", "failed"):
            model_id, chi2_s, fit_s, _k_s, status, rest = m.groups()
        else:
            m = line_re_old.match(line)
            if not m:
                continue
            model_id, chi2_s, fit_s, status, rest = m.groups()
        if status.lower() != "success":
            continue
        try:
            chi2 = float(chi2_s)
            fit = float(fit_s)
        except ValueError:
            continue
        if not np.isfinite(fit) or fit >= best_fit:
            continue
        expr_m = re.search(r"w\(a\)=(.+)$", rest)
        expr = expr_m.group(1).strip() if expr_m else _normalize_expr(rest)
        if not expr:
            continue
        best_fit = fit
        best_chi2 = chi2 if np.isfinite(chi2) else fit
        best_id = model_id
        best_expr = expr

    if best_id is None or best_expr is None:
        raise SystemExit(f"No successful models in {path}")

    print(
        f"Best model from {path.name}: {best_id}  "
        f"fitness={best_fit:.4f}  chi2={best_chi2:.4f}\n"
        f"  expr: {best_expr}",
        flush=True,
    )
    return best_id, best_expr, float(best_chi2)


def _find_model_folder(ga_root: Path, model_id: str) -> Path | None:
    """Locate an existing ``gen_*_model_*`` tree under a GA run directory."""
    direct = ga_root / model_id
    for cand in (direct, direct / model_id):
        if cand.is_dir() and (cand / "class").is_dir():
            return cand
    # Shallow search (avoid walking huge trees)
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


def _looks_like_model_dir(path: Path) -> bool:
    path = path.expanduser()
    if not path.is_dir():
        return False
    if (path / "class").is_dir():
        return True
    if path.name == "class":
        return True
    nested = path / path.name
    return nested.is_dir() and (nested / "class").is_dir()


def _expr_from_summary(model_name: str, search_roots: list[Path]) -> str | None:
    for root in search_roots:
        for path in (
            root / "GA_summary.txt",
            root.parent / "GA_summary.txt",
            root.parent.parent / "GA_summary.txt",
            Path.cwd() / "GA_summary.txt",
        ):
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip().startswith(model_name):
                    continue
                m = re.search(r"w\(a\)=(.+)$", line.strip())
                if m:
                    return m.group(1).strip()
    return None


def _expr_from_user_eq_h(class_dir: Path) -> str | None:
    """Recover a GA-style expr from ``include/user_eq.h`` (no GA_summary needed)."""
    header = class_dir / "include" / "user_eq.h"
    if not header.is_file():
        return None
    body = None
    for line in header.read_text(encoding="utf-8", errors="replace").splitlines():
        if "FLUID_W_FLD" not in line or "#define" not in line:
            continue
        m = re.search(r"#define\s+FLUID_W_FLD\s*\([^)]+\)\s*\((.*)\)\s*$", line.strip())
        if m:
            body = m.group(1)
            break
    if not body:
        return None
    body = body.replace("pba->", "")
    # pow(x, n) → (x)^n  (repeat for nesting)
    for _ in range(8):
        new = re.sub(r"pow\(([^,()]+),\s*(\d+)\)", r"(\1)^\2", body)
        if new == body:
            break
        body = new
    body = re.sub(r"\s+", "", body)
    return body or None


def _expr_from_model_folder(model_root: Path, class_dir: Path, model_name: str) -> str | None:
    """Try expr.txt → GA_summary → user_eq.h."""
    for path in (model_root / "expr.txt", class_dir / "expr.txt"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                return _normalize_expr(text)
    found = _expr_from_summary(model_name, [model_root])
    if found:
        return found
    return _expr_from_user_eq_h(class_dir)


def _normalize_expr(expr: str) -> str:
    expr = (expr or "").strip()
    if expr.startswith("w(a)="):
        expr = expr.split("=", 1)[1].strip()
    elif "=" in expr and expr.lower().startswith("w"):
        expr = expr.split("=", 1)[1].strip()
    return expr


def _newest_client_run(class_dir: Path) -> Path | None:
    base = class_dir / "scripts" / "client_runs"
    if not base.is_dir():
        return None
    runs = [p for p in base.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def check_client_r1_converged(run_dir: Path) -> tuple[bool, float | None, str]:
    """
    Parse CLiENT ``metrics.log`` Gelman–Rubin table.

    Returns (converged, last_R-1, message). Converged only if the *last*
    iteration is marked True (or last R-1 is below 0.01 if the flag is missing).
    """
    metrics = run_dir / "metrics.log"
    if not metrics.is_file():
        return False, None, f"No metrics.log in {run_dir}"

    text = metrics.read_text(encoding="utf-8", errors="replace")
    rows: list[tuple[int, float, bool]] = []
    in_block = False
    for line in text.splitlines():
        if "Convergence Metrics (Gelman-Rubin" in line:
            in_block = True
            continue
        if in_block and line.strip().startswith("Per-Iteration"):
            break
        if not in_block:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3 or parts[0] in ("", "it") or parts[0].startswith("-"):
            continue
        try:
            it = int(parts[0])
            r1 = float(parts[1])
            flag = parts[2].lower().startswith("t")
            rows.append((it, r1, flag))
        except ValueError:
            continue

    if not rows:
        return False, None, "No R-1 rows found in metrics.log"

    it, r1, flag = rows[-1]
    ok = flag or r1 < 0.01
    msg = (
        f"Last CLiENT iteration {it}: R-1={r1:.5f}, converged={flag} "
        f"({'OK' if ok else 'NOT converged — do not use for PoC plot'})"
    )
    return ok, r1, msg


def _rebuild_model(model_name: str, expr: str) -> Path:
    """Fresh CLASS tree for the winner; returns ``.../class`` directory."""
    from run_model_results import (
        build_model_classy,
        compile_class,
        prepare_model_directory,
        update_class_for_new_params,
    )
    from user_generator_GA import UserEquationGenerator

    class_dir = prepare_model_directory(model_name)
    UserEquationGenerator(class_dir, model_name, expr=expr).run()
    update_class_for_new_params(class_dir, expr)
    compile_class(class_dir, model_name)
    build_model_classy(class_dir, model_name)
    return Path(class_dir)


def _sampled_names(yaml_path: Path) -> list[str]:
    import yaml

    info = yaml.safe_load(yaml_path.read_text())
    return [
        name
        for name, spec in (info.get("params") or {}).items()
        if isinstance(spec, dict) and "prior" in spec
    ]


def _training_xy(run_dir: Path):
    import h5py

    xs, ys = [], []
    for path in sorted((run_dir / "training_data").glob("data_it_*.h5")):
        with h5py.File(path, "r") as f:
            xs.append(f["data"]["x"][...])
            ys.append(f["data"]["y"][...])
    if not xs:
        raise SystemExit(f"No data_it_*.h5 in {run_dir / 'training_data'}")
    return np.vstack(xs), np.concatenate(ys)


def _active_w_names(expr: str, sampled: list[str]) -> list[str]:
    found = []
    for raw in re.findall(r"\b(w\d+|wa|w0)\b", expr):
        name = {"w0": "w0_fld", "wa": "wa_fld"}.get(raw, raw)
        if name in sampled and name not in found:
            found.append(name)
    return found


def _latest_chain(run_dir: Path) -> Path:
    files = sorted((run_dir / "training_chains").glob("chain_it_*.h5"))
    if not files:
        raise SystemExit(f"No chain_it_*.h5 in {run_dir / 'training_chains'}")
    return files[-1]


def make_plots(
    model_root: Path,
    class_dir: Path,
    run_dir: Path,
    expr: str,
    *,
    ga_chi2: float | None = None,
    cobaya_chi2: float | None = None,
) -> Path:
    """Write param table + GetDist triangle under ``model_root/post_ga_plots``."""
    import h5py
    from getdist import MCSamples, plots
    from matplotlib import pyplot as plt

    yaml_path = class_dir / "scripts" / "client_config" / f"{model_root.name}.yaml"
    if not yaml_path.is_file():
        # Fallback: any yaml in client_config
        cfgs = list((class_dir / "scripts" / "client_config").glob("*.yaml"))
        cfgs = [p for p in cfgs if not p.name.endswith("_client.yaml")]
        if not cfgs:
            raise SystemExit(f"No Cobaya yaml under {yaml_path.parent}")
        yaml_path = cfgs[0]

    names = _sampled_names(yaml_path)
    x, y = _training_xy(run_dir)
    best_i = int(np.argmax(y))
    best = {n: float(x[best_i, i]) for i, n in enumerate(names)}
    chi2_train = float(-2.0 * y[best_i])
    active_w = _active_w_names(expr, names)

    out = model_root / "post_ga_plots"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n--- post-GA plots ---")
    print(f"run_dir     : {run_dir}")
    print(f"chi2 train  : {chi2_train:.4f}  (-2 * max training logL)")
    if ga_chi2 is not None:
        print(f"chi2 GA     : {ga_chi2:.4f}")
    if cobaya_chi2 is not None:
        print(f"chi2 Cobaya : {cobaya_chi2:.4f}  (MAP validation)")

    txt = out / f"{model_root.name}_params.txt"
    with txt.open("w", encoding="utf-8") as f:
        f.write(f"model: {model_root.name}\n")
        f.write(f"expr: {expr}\n")
        f.write(f"run: {run_dir}\n")
        f.write(f"chi2_training_approx: {chi2_train:.6f}\n")
        if ga_chi2 is not None:
            f.write(f"chi2_ga: {ga_chi2:.6f}\n")
        if cobaya_chi2 is not None:
            f.write(f"chi2_cobaya_map: {cobaya_chi2:.6f}\n")
        f.write("\n=== all sampled parameters (best training point) ===\n")
        for n, v in best.items():
            f.write(f"{n:<16} {v:.8g}\n")
        f.write("\n=== w coefficients in this w(a) ===\n")
        for n in active_w:
            f.write(f"{n:<16} {best[n]:.8g}\n")

    fig, axes = plt.subplots(2, 1, figsize=(8, 0.4 * (len(best) + len(active_w) + 8)))
    for ax in axes:
        ax.axis("off")
    axes[0].set_title("All sampled parameters (best training point)")
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

    chain_path = _latest_chain(run_dir)
    with h5py.File(chain_path, "r") as f:
        chain = np.asarray(f["mcmc"]["chain"][...])
    flat = chain.reshape(-1, chain.shape[-1])
    samples = MCSamples(samples=flat, names=names, labels=names, label="CLiENT T=1")

    triangle_params = [n for n in ("H0", "omega_cdm", "omega_b", "Omega_m") if n in names]
    triangle_params += [n for n in active_w if n in names]
    g = plots.get_subplot_plotter(width_inch=8)
    g.triangle_plot(samples, triangle_params, filled=True, title_limit=1)
    g.fig.suptitle(f"{model_root.name}  (post-GA CLiENT, T=1)", y=1.02)
    triangle_pdf = out / f"{model_root.name}_getdist_triangle.pdf"
    g.fig.savefig(triangle_pdf, bbox_inches="tight")

    print(f"Wrote {txt}")
    print(f"Wrote {table_pdf}")
    print(f"Wrote {triangle_pdf}  (from {chain_path.name})")
    return out


def run_post_ga(
    *,
    expr: str,
    model_dir: str | Path | None = None,
    model_name: str = "post_ga_best",
    profile: str = "ga_final",
    cobaya_config: str | None = None,
    ga_chi2: float | None = None,
    rebuild: bool = False,
    plot: bool = True,
    validate_map: bool = True,
    require_convergence: bool = True,
) -> dict:
    """
    Full post-GA pipeline. Returns dict with chi2, run_dir, plot_dir, etc.

    Parameters
    ----------
    expr
        Winning ``w(a)`` string (with or without ``w(a)=`` prefix).
    model_dir
        Existing ``gen_*_model_*`` folder (contains ``class/``). If None or
        ``rebuild=True``, builds a fresh tree named ``model_name``.
    cobaya_config
        Optional override (e.g. ``sn_desi`` / ``cmb_spa``). If omitted and the
        model folder already has ``client_config/<model>.yaml``, that file's
        likelihoods are kept. Otherwise defaults to ``sn_desi``.
    require_convergence
        If True (default), refuse GetDist plots unless metrics.log shows
        Gelman–Rubin convergence on the last iteration.
    """
    expr = _normalize_expr(expr)
    if not expr:
        raise ValueError("expr is required")

    os.environ["CLIENT_PROFILE"] = profile

    if model_dir is not None and not rebuild:
        _name, model_root, class_dir = _resolve_model_dirs(Path(model_dir))
        model_name = _name
    else:
        print(f"\n=== post-GA: rebuilding CLASS as {model_name!r} ===", flush=True)
        class_dir = _rebuild_model(model_name, expr)
        model_root = class_dir.parent

    folder_yaml = class_dir / "scripts" / "client_config" / f"{model_name}.yaml"
    reuse_yaml = False
    yaml_note = ""

    from cobaya_runner import set_cobaya_yaml

    if cobaya_config:
        yaml_used = set_cobaya_yaml(cobaya_config)
        yaml_note = f"override template: {yaml_used}"
    elif folder_yaml.is_file():
        reuse_yaml = True
        yaml_used = str(folder_yaml)
        yaml_note = f"model folder yaml (likelihoods kept): {yaml_used}"
        # Still register path so Cobaya MAP validation uses the same datasets.
        set_cobaya_yaml(yaml_used)
    else:
        cfg = (
            os.environ.get("GA_COBAYA_CONFIG", "").strip()
            or os.environ.get("GA_COBAYA_YAML", "").strip()
            or "sn_desi"
        )
        yaml_used = set_cobaya_yaml(cfg)
        yaml_note = f"no folder yaml; using {yaml_used}"

    print(f"\n=== post-GA CLiENT ({profile}) ===", flush=True)
    print(f"model_name : {model_name}")
    print(f"class_dir  : {class_dir}")
    print(f"expr       : {expr}")
    print(f"cobaya     : {yaml_note}")
    print(
        "Profile ga_final aims for R-1 < 0.01 with long emcee "
        "(not a fixed 20-iteration cutoff).",
        flush=True,
    )

    from client_runner import run_client_best_chi2
    import inspect

    client_kwargs = dict(
        profile=profile,
        validate_map=validate_map,
        allow_cobaya_fallback=False,  # PoC: never silently switch to short Cobaya MCMC
    )
    # Image may be older than this post_ga.py — only pass if supported.
    sig = inspect.signature(run_client_best_chi2).parameters
    if "reuse_cobaya_yaml" in sig:
        client_kwargs["reuse_cobaya_yaml"] = reuse_yaml
    elif reuse_yaml:
        print(
            "WARNING: image client_runner has no reuse_cobaya_yaml; "
            "will rewrite yaml from COBAYA template.",
            flush=True,
        )
    if "allow_cobaya_fallback" not in sig:
        client_kwargs.pop("allow_cobaya_fallback", None)

    chi2, backend = run_client_best_chi2(
        expr,
        str(class_dir),
        model_name,
        master_classy_dir=None,
        **client_kwargs,
    )
    print(f"post-GA chi2={chi2:.4f}  backend={backend}", flush=True)

    run_dir = _newest_client_run(class_dir)
    plot_dir = None
    r1_ok, r1, r1_msg = False, None, "no run_dir"
    if run_dir is not None:
        r1_ok, r1, r1_msg = check_client_r1_converged(run_dir)
        print(r1_msg, flush=True)

    if plot and run_dir is not None:
        if require_convergence and not r1_ok:
            raise SystemExit(
                f"\nRefusing GetDist PoC plot: CLiENT did not converge.\n"
                f"  {r1_msg}\n"
                f"  Re-run with the updated ga_final profile (longer chains, "
                f"more iterations), or pass --allow-unconverged to plot anyway.\n"
                f"  Run dir: {run_dir}"
            )
        plot_dir = make_plots(
            model_root,
            class_dir,
            run_dir,
            expr,
            ga_chi2=ga_chi2,
            cobaya_chi2=float(chi2) if validate_map else None,
        )
    elif plot:
        print("WARNING: no client_runs/ found; skipped plots", flush=True)

    return {
        "model_name": model_name,
        "model_root": str(model_root),
        "class_dir": str(class_dir),
        "expr": expr,
        "chi2": float(chi2),
        "backend": backend,
        "run_dir": str(run_dir) if run_dir else None,
        "plot_dir": str(plot_dir) if plot_dir else None,
        "ga_chi2": ga_chi2,
        "cobaya_yaml": yaml_used,
        "reuse_cobaya_yaml": reuse_yaml,
        "r1_converged": r1_ok,
        "r1": r1,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Post-GA CLiENT final pass + GetDist. "
            "Picks the best model from GA_summary.txt unless you pass a model folder."
        )
    )
    p.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Either a gen_*_model_* folder, a GA run dir with GA_summary.txt, "
            "or omit to use ./GA_summary.txt"
        ),
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Explicit GA_summary.txt (or its parent dir) to pick the best model",
    )
    p.add_argument("--expr", default=None, help="Override w(a); else from summary/folder")
    p.add_argument(
        "--model-name",
        default=None,
        help="Name when rebuilding (default: winner id from summary, e.g. gen_X_model_Y)",
    )
    p.add_argument(
        "--profile",
        default="ga_final",
        help="CLiENT profile: ga_final (laptop) or ga_final_hpc (long emcee)",
    )
    p.add_argument(
        "--cobaya-config",
        default=None,
        help="Cobaya YAML: sn_desi (Pantheon+SH0ES+DESI) or cmb_spa, or a path.",
    )
    p.add_argument("--ga-chi2", type=float, default=None, help="GA fitness to record")
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a fresh CLASS tree even if a model folder exists",
    )
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-validate-map", action="store_true")
    p.add_argument(
        "--allow-unconverged",
        action="store_true",
        help="Plot even if metrics.log does not show R-1 convergence "
        "(not recommended for a PoC figure).",
    )
    args = p.parse_args(argv)

    model_dir: Path | None = None
    model_name = args.model_name
    expr = _normalize_expr(args.expr or "")
    ga_chi2 = args.ga_chi2
    rebuild = bool(args.rebuild)

    path = args.path.expanduser().resolve() if args.path is not None else None

    # 1) Explicit model folder with class/
    if path is not None and _looks_like_model_dir(path):
        name, root, class_dir = _resolve_model_dirs(path)
        model_dir = root
        model_name = model_name or name
        if not expr:
            expr = _expr_from_model_folder(root, class_dir, name) or ""
            if expr:
                print(f"Recovered expr from model folder:\n  {expr}")

    # 2) GA run dir / summary / cwd → auto-pick best model id + expr
    else:
        summary_arg = args.summary
        search_roots: list[Path] = []
        if path is not None:
            search_roots.append(path)
            if summary_arg is None and (path / "GA_summary.txt").is_file():
                summary_arg = path / "GA_summary.txt"
            elif summary_arg is None and path.name == "GA_summary.txt":
                summary_arg = path
        if not expr or model_name is None or ga_chi2 is None:
            mid, sexpr, schi2 = best_from_ga_summary(
                summary_arg, search_roots=search_roots
            )
            model_name = model_name or mid
            if not expr:
                expr = sexpr
            if ga_chi2 is None:
                ga_chi2 = schi2
        # Prefer existing on-disk tree for that winner if present
        if not rebuild and model_dir is None:
            roots = list(search_roots) + [Path.cwd()]
            if summary_arg is not None:
                sp = summary_arg.expanduser().resolve()
                roots.insert(0, sp.parent if sp.is_file() else sp)
            for root in roots:
                found = _find_model_folder(root, model_name)
                if found is not None:
                    model_dir = found
                    print(f"Using existing model folder: {model_dir}", flush=True)
                    break
        if model_dir is None:
            rebuild = True

    if not expr:
        raise SystemExit(
            "Need --expr, a model folder with user_eq.h / expr.txt, "
            "or a GA_summary.txt with a successful winner"
        )
    if not model_name:
        model_name = "post_ga_best"

    run_post_ga(
        expr=expr,
        model_dir=model_dir,
        model_name=model_name,
        profile=args.profile,
        cobaya_config=args.cobaya_config,
        ga_chi2=ga_chi2,
        rebuild=rebuild,
        plot=not args.no_plot,
        validate_map=not args.no_validate_map,
        require_convergence=not args.allow_unconverged,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
