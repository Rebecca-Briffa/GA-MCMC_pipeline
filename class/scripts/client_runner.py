"""
CLiENT (likelihood emulator) integration for the GA pipeline.

CLiENT trains a neural network on true Cobaya+CLASS likelihood evaluations, then
runs fast emcee MCMC on the surrogate. For GA fitness we take the best *true*
log-likelihood from CLiENT training data (initial Latin-hypercube samples), and
optionally re-evaluate that point with Cobaya for an exact chi2__SN.

Requires a separate install: https://github.com/AarhusCosmology/client_public
Set environment variable CLIENT_ROOT to that repository path.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import yaml

from cobaya_runner import run_cobaya_best_chi2
from client_gpu import (
    apply_client_gpu_env,
    format_gpu_status,
    probe_client_gpu,
    resolve_client_profile,
)

__all__ = [
    "client_root",
    "client_python",
    "write_client_yaml",
    "best_chi2_from_client_run",
    "run_client_best_chi2",
    "log_client_gpu_status",
]

# GA-oriented profile: fewer true evals than paper defaults, one training iteration.
GA_CLIENT_PROFILES: dict[str, dict[str, Any]] = {
    "ga_fast": {
        "data": {
            "scalers": {"parameters": "standard", "targets": "standard"},
            "initial": {
                "n_samples": 400,
                "strategy": "lhs",
                "n_std": 8.0,
            },
            "iterative": {
                "n_candidates": 200,
                "strategy": "adaptive",
                "k_NN": 20,
                "temperature": 7.0,
                "update_freq": 50,
            },
        },
        "model": {"n_layers": 3, "n_neurons": 128, "activation": "alsing"},
        "training": {
            "n_epochs": 250,
            "batch_size": 128,
            "loss": "msre",
            "kappa_sigma": 3,
            "learning_rate": 0.001,
            "val_split": 0.1,
            "patience": 40,
        },
        "sampling": {
            "save_chains": True,
            "temperature": 7.0,
            "method": "emcee",
            "emcee": {
                "n_walkers": 32,
                "max_steps": 8000,
                "burn_in": 500,
                "ess_target": 40,
                "delta_tau_tol": 0.05,
                "chunk_size": 500,
                "ac_thin": 10,
            },
        },
        "convergence": {
            "r_minus_one_threshold": 0.02,
            "max_iterations": 1,
        },
        "use_convergence": False,
        "n_iterations": 1,
    },
    "ga_fast_gpu": {
        # Same as ga_fast but larger batches — better GPU utilisation (TensorFlow/Keras).
        "data": {
            "scalers": {"parameters": "standard", "targets": "standard"},
            "initial": {
                "n_samples": 400,
                "strategy": "lhs",
                "n_std": 8.0,
            },
            "iterative": {
                "n_candidates": 200,
                "strategy": "adaptive",
                "k_NN": 20,
                "temperature": 7.0,
                "update_freq": 50,
            },
        },
        "model": {"n_layers": 3, "n_neurons": 128, "activation": "alsing"},
        "training": {
            "n_epochs": 250,
            "batch_size": 512,
            "loss": "msre",
            "kappa_sigma": 3,
            "learning_rate": 0.001,
            "val_split": 0.1,
            "patience": 40,
        },
        "sampling": {
            "save_chains": True,
            "temperature": 7.0,
            "method": "emcee",
            "emcee": {
                "n_walkers": 32,
                "max_steps": 8000,
                "burn_in": 500,
                "ess_target": 40,
                "delta_tau_tol": 0.05,
                "chunk_size": 500,
                "ac_thin": 10,
            },
        },
        "convergence": {
            "r_minus_one_threshold": 0.02,
            "max_iterations": 1,
        },
        "use_convergence": False,
        "n_iterations": 1,
    },
    "ga_converge": {
        # Standard cosmology mode: iterate until Gelman-Rubin R-1 < threshold
        # (do NOT pass client.py -i; see use_convergence).
        "data": {
            "scalers": {"parameters": "standard", "targets": "standard"},
            "initial": {
                "n_samples": 800,
                "strategy": "lhs",
                "n_std": 8.0,
            },
            "iterative": {
                "n_candidates": 400,
                "strategy": "adaptive",
                "k_NN": 20,
                "temperature": 7.0,
                "update_freq": 50,
            },
        },
        "model": {"n_layers": 5, "n_neurons": 256, "activation": "alsing"},
        "training": {
            "n_epochs": 500,
            "batch_size": 128,
            "loss": "msre",
            "kappa_sigma": 3,
            "learning_rate": 0.0005,
            "val_split": 0.1,
            "patience": 80,
        },
        "sampling": {
            "save_chains": True,
            "temperature": 7.0,
            "method": "emcee",
            "emcee": {
                "n_walkers": 64,
                "max_steps": 20000,
                "burn_in": 1000,
                "ess_target": 50,
                "delta_tau_tol": 0.05,
                "chunk_size": 1000,
                "ac_thin": 10,
            },
        },
        "convergence": {
            "r_minus_one_threshold": 0.01,
            "max_iterations": 20,
        },
        "use_convergence": True,
    },
    "ga_accurate": {
        "data": {
            "scalers": {"parameters": "standard", "targets": "standard"},
            "initial": {
                "n_samples": 1500,
                "strategy": "lhs",
                "n_std": 10.0,
            },
            "iterative": {
                "n_candidates": 500,
                "strategy": "adaptive",
                "k_NN": 20,
                "temperature": 7.0,
                "update_freq": 50,
            },
        },
        "model": {"n_layers": 5, "n_neurons": 256, "activation": "alsing"},
        "training": {
            "n_epochs": 500,
            "batch_size": 128,
            "loss": "msre",
            "kappa_sigma": 3,
            "learning_rate": 0.0005,
            "val_split": 0.1,
            "patience": 80,
        },
        "sampling": {
            "save_chains": True,
            "temperature": 7.0,
            "method": "emcee",
            "emcee": {
                "n_walkers": 64,
                "max_steps": 20000,
                "burn_in": 1000,
                "ess_target": 50,
                "delta_tau_tol": 0.05,
                "chunk_size": 1000,
                "ac_thin": 10,
            },
        },
        "convergence": {
            "r_minus_one_threshold": 0.01,
            "max_iterations": 3,
        },
        "use_convergence": False,
        "n_iterations": 3,
    },
    # Post-GA / proof-of-concept (laptop-safe). 256 walkers × 400k previously
    # OOM'd (client.py exit -9) then fell into cobaya_fallback.
    # Aim: R-1 < 0.01; τ is often ~10^3 so GetDist can still look speckled.
    "ga_final": {
        "data": {
            "scalers": {"parameters": "standard", "targets": "standard"},
            "initial": {
                "n_samples": 2000,
                "strategy": "lhs",
                "n_std": 5.0,
            },
            "iterative": {
                "n_candidates": 600,
                "strategy": "adaptive",
                "k_NN": 20,
                "temperature": 1.0,
                "update_freq": 50,
            },
        },
        "model": {"n_layers": 5, "n_neurons": 256, "activation": "alsing"},
        "training": {
            "n_epochs": 600,
            "batch_size": 128,
            "loss": "msre",
            "kappa_sigma": 3,
            "learning_rate": 0.0005,
            "val_split": 0.1,
            "patience": 100,
        },
        "sampling": {
            "save_chains": True,
            "temperature": 1.0,
            "method": "emcee",
            "emcee": {
                "n_walkers": 64,
                "max_steps": 200000,
                "burn_in": 5000,
                "ess_target": 300,
                "delta_tau_tol": 0.05,
                "chunk_size": 2000,
                "ac_thin": 20,
            },
        },
        "convergence": {
            "r_minus_one_threshold": 0.01,
            "max_iterations": 40,
        },
        "use_convergence": True,
    },
    # Post-GA on HPC / large RAM: longer emcee so GetDist is not ESS-starved
    # (target steps ≫ 50×τ with τ~10^3). Use: post_ga.py ... --profile ga_final_hpc
    "ga_final_hpc": {
        "data": {
            "scalers": {"parameters": "standard", "targets": "standard"},
            "initial": {
                "n_samples": 5000,
                "strategy": "lhs",
                "n_std": 5.0,
            },
            "iterative": {
                "n_candidates": 1200,
                "strategy": "adaptive",
                "k_NN": 20,
                "temperature": 1.0,
                "update_freq": 50,
            },
        },
        "model": {"n_layers": 5, "n_neurons": 256, "activation": "alsing"},
        "training": {
            "n_epochs": 800,
            "batch_size": 128,
            "loss": "msre",
            "kappa_sigma": 3,
            "learning_rate": 0.0005,
            "val_split": 0.1,
            "patience": 120,
        },
        "sampling": {
            "save_chains": True,
            "temperature": 1.0,
            "method": "emcee",
            "emcee": {
                "n_walkers": 128,
                "max_steps": 800000,
                "burn_in": 20000,
                "ess_target": 1000,
                "delta_tau_tol": 0.03,
                "chunk_size": 4000,
                "ac_thin": 50,
            },
        },
        "convergence": {
            "r_minus_one_threshold": 0.01,
            "max_iterations": 60,
        },
        "use_convergence": True,
    },
}


def client_root() -> str:
    root = os.environ.get("CLIENT_ROOT", "").strip()
    if not root:
        raise RuntimeError(
            "CLIENT_ROOT is not set. Clone https://github.com/AarhusCosmology/client_public "
            "and export CLIENT_ROOT=/path/to/client_public"
        )
    root = os.path.abspath(root)
    client_py = os.path.join(root, "client.py")
    if not os.path.isfile(client_py):
        raise RuntimeError(f"CLIENT_ROOT does not contain client.py: {root}")
    return root


def _python_has_sklearn(python_exe: str) -> bool:
    proc = subprocess.run(
        [python_exe, "-c", "import sklearn"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def client_python(prefer: str | None = None) -> str:
    """
    Return the Python interpreter used to run ``client.py``.

    CLiENT needs ``sklearn`` and TensorFlow. Prefer ``GA_CLIENT_PYTHON`` when set,
    otherwise ``sys.executable`` if it can import sklearn, otherwise ``clienv``.
    """
    import shutil

    if prefer:
        exe = os.path.abspath(prefer)
        if not os.path.isfile(exe):
            raise RuntimeError(f"GA_CLIENT_PYTHON is not a file: {prefer!r}")
        if not _python_has_sklearn(exe):
            raise RuntimeError(
                f"GA_CLIENT_PYTHON={exe!r} cannot import sklearn (required by CLiENT)"
            )
        return exe

    env_exe = os.environ.get("GA_CLIENT_PYTHON", "").strip()
    if env_exe:
        return client_python(env_exe)

    current = os.path.abspath(sys.executable)
    if _python_has_sklearn(current):
        return current

    candidates: list[str] = []
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        base = os.path.dirname(conda_prefix)
        candidates.append(os.path.join(base, "clienv", "bin", "python"))
    candidates.extend(
        [
            os.path.expanduser("~/miniconda3/envs/clienv/bin/python"),
            os.path.expanduser("~/anaconda3/envs/clienv/bin/python"),
        ]
    )
    which = shutil.which("python", path=os.path.expanduser("~/miniconda3/envs/clienv/bin"))
    if which:
        candidates.append(which)

    for exe in candidates:
        exe = os.path.abspath(exe)
        if os.path.isfile(exe) and _python_has_sklearn(exe):
            return exe

    raise RuntimeError(
        f"CLiENT requires sklearn but {current} cannot import it. "
        "Activate clienv before running the GA:\n"
        "  conda activate clienv\n"
        "  export GA_LIKELIHOOD=client\n"
        "Or set GA_CLIENT_PYTHON=/path/to/clienv/bin/python"
    )


def log_client_gpu_status(python_exe: str | None = None) -> dict:
    """Probe TensorFlow GPUs and print a one-line status (GA startup)."""
    py = python_exe or client_python()
    probe = probe_client_gpu(py)
    print(format_gpu_status(probe), flush=True)
    return probe


def write_client_yaml(
    cobaya_yaml_path: str,
    client_yaml_path: str,
    profile: str = "ga_fast",
) -> str:
    """Write CLiENT master YAML pointing at a model-specific Cobaya file."""
    if profile not in GA_CLIENT_PROFILES:
        raise ValueError(
            f"Unknown CLIENT profile {profile!r}; "
            f"choose from {list(GA_CLIENT_PROFILES)}"
        )
    prof = GA_CLIENT_PROFILES[profile]
    cfg = {
        "likelihood": {
            "wrapper": "cobaya",
            "param": os.path.abspath(cobaya_yaml_path),
            "conf": None,
            "path": None,
        },
        **{k: prof[k] for k in ("data", "model", "training", "sampling", "convergence")},
    }
    os.makedirs(os.path.dirname(os.path.abspath(client_yaml_path)), exist_ok=True)
    with open(client_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    return client_yaml_path


def _load_training_h5(run_dir: str) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    training_dir = os.path.join(run_dir, "training_data")
    if not os.path.isdir(training_dir):
        raise FileNotFoundError(f"No training_data/ in {run_dir}")

    xs, ys = [], []
    for name in sorted(os.listdir(training_dir)):
        if not name.startswith("data_it_") or not name.endswith(".h5"):
            continue
        path = os.path.join(training_dir, name)
        with h5py.File(path, "r") as f:
            xs.append(f["data"]["x"][...])
            ys.append(f["data"]["y"][...])
    if not xs:
        raise FileNotFoundError(f"No data_it_*.h5 under {training_dir}")
    return np.vstack(xs), np.concatenate(ys)


def _load_param_names(cobaya_yaml_path: str) -> list[str]:
    from cobaya.model import get_model
    from cobaya.yaml import yaml_load_file

    info = yaml_load_file(cobaya_yaml_path)
    model = get_model(info)
    return list(model.parameterization.sampled_params())


def best_chi2_from_client_run(
    run_dir: str,
    cobaya_yaml_path: str,
    *,
    validate_with_cobaya: bool = True,
    expr: str | None = None,
    model_class_dir: str | None = None,
    model_name: str | None = None,
    master_classy_dir: str | None = None,
) -> float:
    """
    Best SN chi2 from CLiENT training data (true likelihood evaluations).

    If ``validate_with_cobaya``, re-run Cobaya once at that parameter vector for
    an exact ``chi2__SN`` (recommended for GA ranking).
    """
    x, y = _load_training_h5(run_dir)
    param_names = _load_param_names(cobaya_yaml_path)
    if x.shape[1] != len(param_names):
        raise ValueError(
            f"Parameter count mismatch: x has {x.shape[1]} cols, "
            f"expected {len(param_names)}"
        )

    best_i = int(np.argmax(y))
    best_params = {name: float(x[best_i, j]) for j, name in enumerate(param_names)}
    chi2_approx = float(-2.0 * y[best_i])

    if not validate_with_cobaya:
        return chi2_approx

    if not all([expr, model_class_dir, model_name]):
        raise ValueError(
            "validate_with_cobaya requires expr, model_class_dir, and model_name"
        )

    from cobaya_runner import chi2_sn_at_params

    return chi2_sn_at_params(
        expr,
        model_class_dir,
        model_name,
        best_params,
        master_classy_dir=master_classy_dir,
    )


def run_client_best_chi2(
    expr,
    model_class_dir,
    model_name,
    master_classy_dir=None,
    *,
    profile: str | None = None,
    validate_map: bool | None = None,
    reuse_cobaya_yaml: bool = False,
    allow_cobaya_fallback: bool = True,
):
    """
    Run CLiENT for one GA model.

    Returns (chi2, likelihood_backend) with backend ``client`` or
    ``cobaya_fallback`` if CLiENT is missing or fails.

    If ``reuse_cobaya_yaml``, keep likelihoods from the model folder's existing
    ``client_config/<model>.yaml`` (only refresh classy path + params).

    If ``allow_cobaya_fallback`` is False (recommended for post-GA PoC), raise
    instead of starting a short Cobaya MCMC when CLiENT dies (e.g. OOM / -9).
    """
    profile = resolve_client_profile(profile)
    if validate_map is None:
        validate_map = os.environ.get("CLIENT_VALIDATE_MAP", "1") != "0"

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(scripts_dir, "client_worker.py")

    def _fail_or_fallback(msg: str, stderr_tail: str | None = None):
        if stderr_tail:
            print(f"[{model_name}] CLiENT stderr (tail):\n{stderr_tail}", flush=True)
        if "exited with code -9" in msg or "code -9" in msg:
            print(
                f"[{model_name}] Hint: exit -9 is usually the Linux OOM killer "
                "(emcee chain too large for RAM). ga_final was reduced for laptops; "
                "close other apps or lower n_walkers/max_steps further.",
                flush=True,
            )
        if not allow_cobaya_fallback:
            raise RuntimeError(
                f"CLiENT failed and Cobaya fallback is disabled (post-GA PoC): {msg}"
            )
        print(
            f"[{model_name}] {msg}; falling back to Cobaya MCMC",
            flush=True,
        )
        chi2 = run_cobaya_best_chi2(
            expr, model_class_dir, model_name, master_classy_dir
        )
        print(f"[{model_name}] likelihood_backend: cobaya_fallback", flush=True)
        return chi2, "cobaya_fallback", {"run_dir": None}

    with tempfile.TemporaryDirectory(prefix="client_job_") as tmp:
        job_path = os.path.join(tmp, "job.json")
        result_path = os.path.join(tmp, "result.json")
        job = {
            "expr": expr,
            "model_class_dir": model_class_dir,
            "model_name": model_name,
            "master_classy_dir": master_classy_dir,
            "profile": profile,
            "validate_map": validate_map,
            "reuse_cobaya_yaml": reuse_cobaya_yaml,
        }
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f)

        print(f"\n=== CLiENT: {model_name} (profile={profile}) ===", flush=True)
        worker_env = apply_client_gpu_env(os.environ.copy())
        proc = subprocess.run(
            [sys.executable, worker, job_path, result_path],
            cwd=scripts_dir,
            env=worker_env,
        )

        if not os.path.isfile(result_path):
            return _fail_or_fallback(
                f"CLiENT worker failed (exit {proc.returncode})"
            )

        with open(result_path, encoding="utf-8") as f:
            payload = json.load(f)

    if payload.get("status") != "ok":
        err = payload.get("error", "unknown")
        return _fail_or_fallback(
            f"CLiENT error: {err}",
            payload.get("stderr_tail"),
        )

    if payload.get("warning"):
        print(f"[{model_name}] {payload['warning']}", flush=True)
    if payload.get("gpu_status"):
        print(f"[{model_name}] CLiENT GPU: {payload['gpu_status']}", flush=True)
    print(f"[{model_name}] likelihood_backend: client", flush=True)
    # Third element: optional meta (run_dir for DIC / constraint checks).
    meta = {"run_dir": payload.get("run_dir")}
    return float(payload["chi2"]), "client", meta
