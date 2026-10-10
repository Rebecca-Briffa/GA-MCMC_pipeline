"""Resolve per-model CLASSy build paths (no Cobaya imports)."""
from __future__ import annotations

import os
import sys


def find_model_classy_lib_dir(model_class_dir: str) -> str:
    """Return python/build/lib.* containing the model's classy.so."""
    model_python = os.path.join(model_class_dir, "python")
    build_dir = os.path.join(model_python, "build")
    if not os.path.isdir(build_dir):
        raise FileNotFoundError(
            f"No CLASSy build under {model_python}. "
            "Run build_model_classy() before Cobaya/CLiENT."
        )
    lib_dirs = sorted(
        d
        for d in os.listdir(build_dir)
        if d.startswith("lib.") and os.path.isdir(os.path.join(build_dir, d))
    )
    if not lib_dirs:
        raise FileNotFoundError(
            f"No build/lib.* under {model_python} after CLASSy build."
        )
    return os.path.join(model_python, "build", lib_dirs[-1])


def prepend_model_classy_path(model_class_dir: str) -> str:
    """
    Prefer the model-specific classy.so over a global pip install in clienv.

    Must run before importing cobaya or classy in the worker process.
    """
    lib_dir = os.path.abspath(find_model_classy_lib_dir(model_class_dir))
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    old = os.environ.get("PYTHONPATH", "")
    parts = [p for p in old.split(":") if p]
    if lib_dir not in parts:
        os.environ["PYTHONPATH"] = lib_dir + (":" + old if old else "")
    return lib_dir
