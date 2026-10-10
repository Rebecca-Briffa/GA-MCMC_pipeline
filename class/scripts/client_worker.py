#!/usr/bin/env python3
"""
Standalone CLiENT worker for the GA pipeline.

    python client_worker.py <job.json> <result.json>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback

from client_gpu import apply_client_gpu_env, format_gpu_status, probe_client_gpu

os.environ.setdefault("COBAYA_NOMPI", "1")


def _newest_run_dir(output_base: str, before: set[str]) -> str | None:
    after = set(os.listdir(output_base)) - before
    candidates = [
        os.path.join(output_base, d)
        for d in after
        if os.path.isdir(os.path.join(output_base, d))
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: client_worker.py <job.json> <result.json>", file=sys.stderr)
        return 2

    job_path, result_path = sys.argv[1], sys.argv[2]
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)

    model_name = job["model_name"]
    model_class_dir = job["model_class_dir"]
    expr = job["expr"]
    profile = job.get("profile", "ga_fast")
    validate_map = job.get("validate_map", True)

    from classy_path import find_model_classy_lib_dir, prepend_model_classy_path

    prepend_model_classy_path(model_class_dir)

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    def write_result(payload: dict, code: int = 0) -> int:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return code

    try:
        from client_runner import (
            GA_CLIENT_PROFILES,
            best_chi2_from_client_run,
            client_python,
            client_root,
            write_client_yaml,
        )
        from cobaya_runner import (
            clear_cached_classy_modules,
            refresh_existing_cobaya_yaml,
            write_cobaya_yaml,
        )

        clear_cached_classy_modules()

        root = client_root()
        profile_cfg = GA_CLIENT_PROFILES[profile]
        use_convergence = profile_cfg.get("use_convergence", False)
        if os.environ.get("CLIENT_USE_CONVERGENCE", "").strip() in ("1", "true", "yes"):
            use_convergence = True
        if os.environ.get("CLIENT_USE_CONVERGENCE", "").strip() in ("0", "false", "no"):
            use_convergence = False

        config_dir = os.path.join(model_class_dir, "scripts", "client_config")
        cobaya_yaml = os.path.join(config_dir, f"{model_name}.yaml")
        client_yaml = os.path.join(config_dir, f"{model_name}_client.yaml")
        reuse = bool(job.get("reuse_cobaya_yaml")) and os.path.isfile(cobaya_yaml)

        if reuse:
            refresh_existing_cobaya_yaml(
                expr,
                model_class_dir,
                model_name,
                cobaya_yaml,
                job.get("master_classy_dir"),
            )
        else:
            write_cobaya_yaml(
                expr,
                model_class_dir,
                model_name,
                cobaya_yaml,
                job.get("master_classy_dir"),
            )
        write_client_yaml(cobaya_yaml, client_yaml, profile=profile)

        output_base = os.path.join(model_class_dir, "scripts", "client_runs")
        os.makedirs(output_base, exist_ok=True)
        before = set(os.listdir(output_base))

        client_py = os.path.join(root, "client.py")
        py = client_python()
        gpu_probe = probe_client_gpu(py)
        gpu_status = format_gpu_status(gpu_probe)
        print(gpu_status, flush=True)

        cmd = [
            py,
            client_py,
            os.path.abspath(client_yaml),
            "-n",
            model_name,
            "-o",
            output_base,
        ]
        if use_convergence:
            print(
                "CLiENT mode: Gelman-Rubin convergence "
                f"(R-1 < {profile_cfg['convergence']['r_minus_one_threshold']}, "
                f"max_iterations={profile_cfg['convergence']['max_iterations']})",
                flush=True,
            )
        else:
            n_it = profile_cfg.get("n_iterations", 1)
            cmd.extend(["-i", str(n_it)])
            print(f"CLiENT mode: fixed iterations (-i {n_it})", flush=True)
        print(f"Running: {' '.join(cmd)}\n", flush=True)

        client_env = apply_client_gpu_env(os.environ.copy())
        try:
            classy_lib = find_model_classy_lib_dir(model_class_dir)
            old_pp = client_env.get("PYTHONPATH", "")
            client_env["PYTHONPATH"] = classy_lib + (":" + old_pp if old_pp else "")
        except FileNotFoundError:
            pass

        proc = subprocess.run(cmd, cwd=root, env=client_env, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout, end="", flush=True)
        if proc.stderr:
            print(proc.stderr, end="", flush=True)
        if proc.returncode != 0:
            return write_result(
                {
                    "status": "error",
                    "error": f"client.py exited with code {proc.returncode}",
                    "client_python": py,
                    "gpu_status": gpu_status,
                    "stderr_tail": (proc.stderr or "")[-4000:],
                },
                1,
            )

        run_dir = _newest_run_dir(output_base, before)
        if run_dir is None:
            return write_result(
                {
                    "status": "error",
                    "error": "No CLiENT output directory created",
                    "gpu_status": gpu_status,
                },
                1,
            )

        chi2 = best_chi2_from_client_run(
            run_dir,
            cobaya_yaml,
            validate_with_cobaya=validate_map,
            expr=expr,
            model_class_dir=model_class_dir,
            model_name=model_name,
            master_classy_dir=job.get("master_classy_dir"),
        )

        return write_result(
            {
                "status": "ok",
                "chi2": chi2,
                "run_dir": run_dir,
                "profile": profile,
                "client_mode": "convergence" if use_convergence else "fixed_iterations",
                "client_python": py,
                "gpu_status": gpu_status,
                "warning": (
                    None
                    if validate_map
                    else "chi2 from training loglkl only (no Cobaya validation)"
                ),
            }
        )

    except Exception as exc:
        return write_result(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
            1,
        )


if __name__ == "__main__":
    raise SystemExit(main())
