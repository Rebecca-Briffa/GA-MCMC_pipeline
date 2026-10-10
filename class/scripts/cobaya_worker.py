#!/usr/bin/env python3
"""
Standalone Cobaya worker for the GA pipeline.

Invoked as:
    python cobaya_worker.py <job.json> <result.json>

Keeps a fresh Python interpreter per model so each classy.so loads cleanly.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

os.environ.setdefault("COBAYA_NOMPI", "1")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: cobaya_worker.py <job.json> <result.json>", file=sys.stderr)
        return 2

    job_path, result_path = sys.argv[1], sys.argv[2]
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)

    from classy_path import prepend_model_classy_path

    prepend_model_classy_path(job["model_class_dir"])

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    from cobaya_runner import (
        _run_cobaya_in_process,
        best_chi2_from_chain,
        best_chi2_from_sample_data,
        clear_cached_classy_modules,
    )

    clear_cached_classy_modules()

    model_name = job["model_name"]
    model_class_dir = job["model_class_dir"]
    print(f"\n=== Cobaya worker: {model_name} ===\n", flush=True)

    def _chi2_from_sampler(sampler):
        products = sampler.products(to_getdist=False)
        sample = products["sample"]
        return best_chi2_from_sample_data(sample.data)

    try:
        _, sampler = _run_cobaya_in_process(
            job["expr"],
            model_class_dir,
            model_name,
            job.get("master_classy_dir"),
        )
        chi2 = _chi2_from_sampler(sampler)
        payload = {"status": "ok", "chi2": chi2}
        exit_code = 0
    except Exception as exc:
        recovered = best_chi2_from_chain(model_class_dir, model_name)
        if recovered is not None:
            payload = {
                "status": "ok",
                "chi2": recovered,
                "warning": (
                    f"Cobaya stopped early ({type(exc).__name__}); "
                    f"used best chi2 from saved chain."
                ),
            }
            exit_code = 0
        else:
            payload = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            exit_code = 1

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
