#!/usr/bin/env python3
"""
Standalone fluid ΛCDM baseline with CLiENT (SN Pantheon+SH0ES + DESI DR2).

Builds CLASS with ``FLUID_W_FLD ≡ -1`` (no free DE coeffs), runs CLiENT, and
writes ``lcdm_summary.txt`` with χ² / k / BIC for ΔBIC plots vs the GA.

Usage (inside ga-wdm pod, after copying this folder + updated params.py)::

    export GA_PROJECT_ROOT=/mnt/lcdm-baseline
    export GA_OUTPUT_DIR=/mnt/lcdm-baseline
    export GA_COBAYA_CONFIG=sn_desi
    export CLIENT_PROFILE=ga_converge   # or ga_fast for a quicker scout
    export GA_LIKELIHOOD=client
    mkdir -p "$GA_PROJECT_ROOT"
    cd /opt/ga-project/class/scripts
    python /mnt/lcdm-baseline/run_lcdm_client.py

Or from this repo tree (dev)::

    cd lcdm_baseline
    python run_lcdm_client.py --profile ga_fast
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow importing GA scripts whether we live next to the repo or on the PVC.
_HERE = Path(__file__).resolve().parent
_REPO_SCRIPTS = _HERE.parent / "class" / "scripts"
_IMAGE_SCRIPTS = Path("/opt/ga-project/class/scripts")
for candidate in (_REPO_SCRIPTS, _IMAGE_SCRIPTS, Path.cwd()):
    if (candidate / "client_runner.py").is_file():
        sys.path.insert(0, str(candidate))
        break
else:
    raise SystemExit(
        "Cannot find class/scripts (client_runner.py). "
        "Run from the GA image or the GA_Github_0708 tree."
    )

os.environ.setdefault("COBAYA_NOMPI", "1")
os.environ.setdefault("GA_COBAYA_CONFIG", "sn_desi")
os.environ.setdefault("GA_LIKELIHOOD", "client")
os.environ.setdefault("CLIENT_PROFILE", "ga_converge")

# Sentinel understood by params.build_ga_params / UserEquationGenerator.
LCDM_EXPR = "-1"
MODEL_NAME = "lcdm_baseline"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLiENT ΛCDM baseline (SN+DESI)")
    p.add_argument(
        "--profile",
        default=os.environ.get("CLIENT_PROFILE", "ga_converge"),
        help="CLiENT profile (default: ga_converge)",
    )
    p.add_argument(
        "--out-dir",
        default=os.environ.get(
            "GA_OUTPUT_DIR",
            os.environ.get("GA_PROJECT_ROOT", str(_HERE / "output")),
        ),
        help="Where to write the model tree + summary",
    )
    p.add_argument(
        "--cobaya-config",
        default=os.environ.get("GA_COBAYA_CONFIG", "sn_desi"),
        help="Cobaya YAML key or path (default: sn_desi)",
    )
    p.add_argument(
        "--no-validate-map",
        action="store_true",
        help="Skip Cobaya MAP re-eval of best CLiENT training point",
    )
    p.add_argument(
        "--allow-cobaya-fallback",
        action="store_true",
        help="If CLiENT fails, fall back to short Cobaya MCMC",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a fresh CLASS tree even if one already exists",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    os.environ["GA_PROJECT_ROOT"] = str(out_root)
    os.environ["GA_OUTPUT_DIR"] = str(out_root)
    os.environ["CLIENT_PROFILE"] = args.profile
    os.environ["GA_COBAYA_CONFIG"] = args.cobaya_config

    from cobaya_runner import set_cobaya_yaml
    from fitness_metrics import count_free_params, n_data, score_from_chi2
    from params import is_lcdm_expr
    from post_ga import _rebuild_model, _resolve_model_dirs
    from client_runner import run_client_best_chi2

    assert is_lcdm_expr(LCDM_EXPR), "LCDM sentinel not recognized by params.py"

    yaml_used = set_cobaya_yaml(args.cobaya_config)
    model_dir = out_root / MODEL_NAME
    if args.rebuild or not model_dir.is_dir():
        print(f"\n=== Building CLASS fluid ΛCDM (expr={LCDM_EXPR!r}) ===", flush=True)
        class_dir = _rebuild_model(MODEL_NAME, LCDM_EXPR)
        model_root = class_dir.parent
    else:
        print(f"\n=== Reusing existing model tree: {model_dir} ===", flush=True)
        _name, model_root, class_dir = _resolve_model_dirs(model_dir)

    k = count_free_params(LCDM_EXPR)
    print(f"model_name : {MODEL_NAME}", flush=True)
    print(f"class_dir  : {class_dir}", flush=True)
    print(f"cobaya     : {yaml_used}", flush=True)
    print(f"profile    : {args.profile}", flush=True)
    print(f"free params k = {k}  (ΛCDM: no free w_i)", flush=True)
    print(f"BIC N_data = {n_data():.0f}", flush=True)

    chi2, backend = run_client_best_chi2(
        LCDM_EXPR,
        str(class_dir),
        MODEL_NAME,
        master_classy_dir=None,
        profile=args.profile,
        validate_map=not args.no_validate_map,
        allow_cobaya_fallback=args.allow_cobaya_fallback,
    )

    fitness, chi2_f, k_out = score_from_chi2(float(chi2), LCDM_EXPR)
    summary = {
        "model": "ΛCDM (fluid w≡-1)",
        "expr": LCDM_EXPR,
        "chi2": chi2_f,
        "k": k_out,
        "bic": fitness,
        "N": n_data(),
        "likelihood_backend": backend,
        "cobaya_config": args.cobaya_config,
        "client_profile": args.profile,
        "class_dir": str(class_dir),
        "model_root": str(model_root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    summary_json = out_root / "lcdm_summary.json"
    summary_txt = out_root / "lcdm_summary.txt"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary_txt.write_text(
        "\n".join(
            [
                "# Fluid ΛCDM baseline (SN Pantheon+SH0ES + DESI DR2)",
                f"# BIC = chi2 + k*ln(N), N={n_data():.0f}",
                f"model     ΛCDM (w≡-1)",
                f"expr      {LCDM_EXPR}",
                f"chi2      {chi2_f:.4f}",
                f"k         {k_out}",
                f"bic       {fitness:.4f}",
                f"backend   {backend}",
                f"profile   {args.profile}",
                f"cobaya    {args.cobaya_config}",
                f"dir       {model_root}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("\n=== ΛCDM baseline done ===", flush=True)
    print(f"chi2={chi2_f:.4f}  k={k_out}  BIC={fitness:.4f}  backend={backend}", flush=True)
    print(f"Wrote {summary_txt}", flush=True)
    print(f"Wrote {summary_json}", flush=True)
    print(
        f"\nFor ΔBIC plots: ΔBIC = BIC_model - {fitness:.4f}",
        flush=True,
    )
    return 0 if math.isfinite(chi2_f) else 1


if __name__ == "__main__":
    raise SystemExit(main())
