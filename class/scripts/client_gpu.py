"""
GPU configuration for CLiENT (TensorFlow) in the GA pipeline.

CLiENT trains Keras models with TensorFlow. When ``tensorflow[and-cuda]`` is
installed in ``clienv``, TF auto-detects NVIDIA GPUs. This module sets
environment variables passed to ``client.py`` and provides a probe utility.

Environment variables (read by the GA pipeline)
-------------------------------------------------
GA_CLIENT_GPU
    ``auto`` (default): use GPU if TensorFlow reports one; else CPU.
    ``1`` / ``force``: require GPU (warn if probe finds none).
    ``0`` / ``cpu``: hide GPUs from TensorFlow (``CUDA_VISIBLE_DEVICES=-1``).

CUDA_VISIBLE_DEVICES
    If already set by the user, it is preserved unless ``GA_CLIENT_GPU=0``.

CLIENT_TF_GPU_ALLOW_GROWTH
    Default ``1``. Sets ``TF_FORCE_GPU_ALLOW_GROWTH=true`` (recommended on
    laptop GPUs with limited VRAM, e.g. RTX 4060 Mobile).

CLIENT_TF_LOG_LEVEL
    Default ``2`` (less TensorFlow stderr spam). Set to ``0`` for full TF logs.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any

__all__ = [
    "client_gpu_mode",
    "apply_client_gpu_env",
    "probe_client_gpu",
    "format_gpu_status",
    "resolve_client_profile",
]

_TRUE = frozenset({"1", "true", "yes", "on", "force", "gpu"})
_FALSE = frozenset({"0", "false", "no", "off", "cpu", "disable"})


def client_gpu_mode() -> str:
    """Return ``auto``, ``force``, or ``cpu``."""
    raw = os.environ.get("GA_CLIENT_GPU", "auto").strip().lower()
    if raw in _TRUE:
        return "force"
    if raw in _FALSE:
        return "cpu"
    return "auto"


def apply_client_gpu_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """
    Return *env* updated with TensorFlow/CUDA settings for CLiENT subprocesses.

    Does not import TensorFlow in the current process.
    """
    out = dict(os.environ if env is None else env)
    out.setdefault("COBAYA_NOMPI", "1")

    log_level = os.environ.get("CLIENT_TF_LOG_LEVEL", "2").strip()
    if log_level:
        out["TF_CPP_MIN_LOG_LEVEL"] = log_level

    if os.environ.get("CLIENT_TF_GPU_ALLOW_GROWTH", "1").strip().lower() not in _FALSE:
        out["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    mode = client_gpu_mode()
    if mode == "cpu":
        out["CUDA_VISIBLE_DEVICES"] = "-1"
        out["GA_CLIENT_GPU_EFFECTIVE"] = "cpu"
        return out

    if mode == "force":
        out["GA_CLIENT_GPU_EFFECTIVE"] = "force"
        # User may still set CUDA_VISIBLE_DEVICES to pick a card.
        return out

    out["GA_CLIENT_GPU_EFFECTIVE"] = "auto"
    return out


def probe_client_gpu(python_exe: str, timeout: float = 120.0) -> dict[str, Any]:
    """
    Run a short TensorFlow probe in *python_exe* (subprocess).

    Returns a dict with keys ``ok``, ``gpus``, ``tensorflow_version``, ``error``.
    """
    script = r"""
import json, os, sys
out = {"ok": False, "gpus": [], "tensorflow_version": None, "error": None,
       "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
try:
    import tensorflow as tf
    out["tensorflow_version"] = tf.__version__
    listed = tf.config.list_physical_devices("GPU")
    out["gpus"] = [d.name for d in listed]
    out["ok"] = True
    if listed:
        for dev in listed:
            try:
                tf.config.experimental.set_memory_growth(dev, True)
            except Exception:
                pass
except Exception as exc:
    out["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out))
"""
    env = apply_client_gpu_env()
    try:
        proc = subprocess.run(
            [python_exe, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "gpus": [], "tensorflow_version": None,
                "error": "probe timed out"}

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[-2000:]
        return {"ok": False, "gpus": [], "tensorflow_version": None, "error": err}

    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"ok": False, "gpus": [], "tensorflow_version": None,
                "error": f"invalid probe output: {line[:500]}"}


def format_gpu_status(probe: dict[str, Any]) -> str:
    """One-line summary for GA logs."""
    if not probe.get("ok"):
        return f"TensorFlow GPU probe failed: {probe.get('error', 'unknown')}"
    tf_ver = probe.get("tensorflow_version") or "?"
    gpus = probe.get("gpus") or []
    mode = os.environ.get("GA_CLIENT_GPU_EFFECTIVE") or client_gpu_mode()
    if gpus:
        names = ", ".join(gpus)
        return f"TensorFlow {tf_ver} | GPU mode={mode} | devices: {names}"
    if mode == "force":
        return (
            f"TensorFlow {tf_ver} | GA_CLIENT_GPU=force but no GPU detected "
            f"(CUDA_VISIBLE_DEVICES={probe.get('cuda_visible_devices')!r})"
        )
    return f"TensorFlow {tf_ver} | no GPU detected — CLiENT will use CPU"


def resolve_client_profile(requested: str | None) -> str:
    """
    Pick CLiENT profile, optionally upgrading ``ga_fast`` → ``ga_fast_gpu``.

    Set ``GA_CLIENT_AUTO_GPU_PROFILE=0`` to disable auto-upgrade.
    """
    profile = (requested or os.environ.get("CLIENT_PROFILE", "ga_fast")).strip()
    if profile != "ga_fast":
        return profile
    if os.environ.get("GA_CLIENT_AUTO_GPU_PROFILE", "1").strip().lower() in _FALSE:
        return profile
    if client_gpu_mode() == "cpu":
        return profile
    return os.environ.get("CLIENT_GPU_PROFILE", "ga_fast_gpu")
