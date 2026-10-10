#!/usr/bin/env python3
"""
Check CLiENT GPU / TensorFlow setup for the GA pipeline.

Usage:
    conda activate clienv
    export CLIENT_ROOT=/home/rebecca/CLASS/client_public
    python check_client_gpu.py
"""
from __future__ import annotations

import os
import sys

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from client_gpu import apply_client_gpu_env, client_gpu_mode, format_gpu_status, probe_client_gpu
from client_runner import client_python, client_root


def main() -> int:
    print("=== CLiENT GPU check ===\n")
    print(f"GA_CLIENT_GPU mode     : {client_gpu_mode()}")
    print(f"CLIENT_ROOT            : {os.environ.get('CLIENT_ROOT', '(not set)')}")
    try:
        root = client_root()
        print(f"CLIENT_ROOT (resolved) : {root}")
    except RuntimeError as exc:
        print(f"CLIENT_ROOT error      : {exc}")
        return 1

    try:
        py = client_python()
    except RuntimeError as exc:
        print(f"\nPython interpreter     : ERROR\n  {exc}")
        return 1

    print(f"Python interpreter     : {py}")

    env = apply_client_gpu_env()
    print(f"CUDA_VISIBLE_DEVICES   : {env.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
    print(f"TF_FORCE_GPU_ALLOW_GROWTH: {env.get('TF_FORCE_GPU_ALLOW_GROWTH', '(unset)')}")
    print(f"TF_CPP_MIN_LOG_LEVEL   : {env.get('TF_CPP_MIN_LOG_LEVEL', '(unset)')}")

    print("\nProbing TensorFlow (may take ~30s on first import)...")
    probe = probe_client_gpu(py)
    print(format_gpu_status(probe))

    if probe.get("gpus"):
        print("\nOK: GPU(s) visible to TensorFlow — CLiENT NN training can use the GPU.")
        print("Recommended GA run block:")
        print("  export GA_LIKELIHOOD=client")
        print("  export CLIENT_PROFILE=ga_fast_gpu   # or ga_fast (auto-upgrades if GPU found)")
        print("  export GA_CLIENT_GPU=auto")
        return 0

    if client_gpu_mode() == "force":
        print("\nWARNING: GA_CLIENT_GPU=force but no GPU detected.")
        return 2

    print("\nNote: CLiENT will still work on CPU; only NN training/sampling is slower.")
    print("CLASS likelihood calls (400 LHS samples) remain CPU-bound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
