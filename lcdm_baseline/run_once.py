#!/usr/bin/env python3
"""
Run CLiENT once for fluid ΛCDM (SN+DESI), then write χ² / k / BIC.

No GA. No GA profiles. Just:
  python client.py client_lcdm.yaml -i 1
  → best true-likelihood χ² from training_data/
  → BIC = chi2 + k*ln(N)

Usage:
  conda activate clienv
  cd lcdm_baseline
  python run_once.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLIENT_ROOT = Path(os.environ.get("CLIENT_ROOT", "/home/rebecca/CLASS/client_public"))
OUT = HERE / "output"
N_DATA = float(os.environ.get("GA_BIC_N_DATA", "1600"))
# Free cosmology params in cobaya_lcdm_sn_desi.yaml (no free w_i)
K_LCDM = 6


def _best_chi2_from_run(run_dir: Path) -> float:
    import h5py
    import numpy as np

    training = run_dir / "training_data"
    xs, ys = [], []
    for path in sorted(training.glob("data_it_*.h5")):
        with h5py.File(path, "r") as f:
            ys.append(f["data"]["y"][...])
    if not ys:
        raise FileNotFoundError(f"No data_it_*.h5 under {training}")
    y = np.concatenate(ys)
    # y is log-likelihood; chi2 ≈ -2 ln L
    return float(-2.0 * np.max(y))


def main() -> int:
    os.environ.setdefault("COBAYA_NOMPI", "1")
    os.environ.setdefault(
        "COBAYA_PACKAGES_PATH",
        "/home/rebecca/Project_hiclass/cobaya_packages",
    )
    # Force CPU if no usable GPU (WSL often blocks NVML)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    client_py = CLIENT_ROOT / "client.py"
    if not client_py.is_file():
        raise SystemExit(f"client.py not found at {client_py}")

    cobaya_yaml = HERE / "cobaya_lcdm_sn_desi.yaml"
    client_yaml = HERE / "client_lcdm.yaml"
    # Rewrite param path to absolute so client can be started from CLIENT_ROOT
    cfg_text = client_yaml.read_text(encoding="utf-8")
    cfg_text = cfg_text.replace(
        "param: cobaya_lcdm_sn_desi.yaml",
        f"param: {cobaya_yaml}",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    resolved_cfg = OUT / "client_lcdm_resolved.yaml"
    resolved_cfg.write_text(cfg_text, encoding="utf-8")

    cmd = [
        sys.executable,
        str(client_py),
        str(resolved_cfg),
        "-n",
        "lcdm",
        "-o",
        str(OUT / "client_runs"),
        "-i",
        "1",
    ]
    print("Running (CLiENT once, ΛCDM only):", flush=True)
    print(" ", " ".join(cmd), flush=True)

    log_path = OUT / "lcdm_run.log"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"cmd: {' '.join(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(CLIENT_ROOT),
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        print(f"CLiENT failed (exit {proc.returncode}). See {log_path}", flush=True)
        return proc.returncode

    runs = sorted(
        (OUT / "client_runs").glob("*"),
        key=lambda p: p.stat().st_mtime,
    )
    runs = [p for p in runs if p.is_dir() and (p / "training_data").is_dir()]
    if not runs:
        print(f"No CLiENT run dir under {OUT / 'client_runs'}", flush=True)
        return 1
    run_dir = runs[-1]

    chi2 = _best_chi2_from_run(run_dir)
    bic = chi2 + K_LCDM * math.log(N_DATA)
    summary = {
        "model": "ΛCDM (w0_fld=-1, wa_fld=0 fixed)",
        "chi2": chi2,
        "k": K_LCDM,
        "N": N_DATA,
        "bic": bic,
        "run_dir": str(run_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "lcdm_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "lcdm_summary.txt").write_text(
        "\n".join(
            [
                "# Fluid ΛCDM — single CLiENT run (SN+DESI)",
                f"chi2      {chi2:.4f}",
                f"k         {K_LCDM}",
                f"N         {N_DATA:.0f}",
                f"bic       {bic:.4f}",
                f"run_dir   {run_dir}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"chi2={chi2:.4f}  k={K_LCDM}  BIC={bic:.4f}", flush=True)
    print(f"Wrote {OUT / 'lcdm_summary.txt'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
