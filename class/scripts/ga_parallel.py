"""
Parallel evaluation of GA models (CLASS compile + likelihood).

Set ``GA_PARALLEL_WORKERS`` (default 1) to evaluate multiple models at once.
Optional ``GA_GPU_IDS=0,1`` assigns one GPU per parallel job (round-robin).

Uses spawn (not fork) so TensorFlow/CUDA is safe in worker processes.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Any, Callable

__all__ = [
    "parallel_workers",
    "gpu_ids_from_env",
    "cuda_device_for_slot",
    "evaluate_jobs_parallel",
    "make_parallel_cal_pop_fitness",
]


def parallel_workers() -> int:
    try:
        return max(1, int(os.environ.get("GA_PARALLEL_WORKERS", "1")))
    except ValueError:
        return 1


def gpu_ids_from_env() -> list[str]:
    raw = os.environ.get("GA_GPU_IDS", "").strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def cuda_device_for_slot(slot: int, gpu_ids: list[str]) -> str | None:
    if not gpu_ids:
        return None
    return gpu_ids[slot % len(gpu_ids)]


def _apply_worker_env(cuda_device: str | None) -> None:
    os.environ.setdefault("COBAYA_NOMPI", "1")
    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    from client_gpu import apply_client_gpu_env

    apply_client_gpu_env(os.environ)


def _evaluate_job(job: dict[str, Any]) -> dict[str, Any]:
    """Top-level worker entry (picklable). Returns job dict + fitness fields."""
    _apply_worker_env(job.get("cuda_device"))

    from GA_MCMC_new import (
        FAILED_FITNESS,
        apply_fitness_metric,
        evaluate_model_likelihood,
        normalize_fitness_value,
    )
    from fitness_metrics import count_free_params
    from posterior_diagnostics import require_constrained

    raw_chi2, likelihood_backend, meta = evaluate_model_likelihood(
        job["expr"],
        job["model_name"],
    )
    chi2 = normalize_fitness_value(raw_chi2)
    meta = meta or {}
    fail_reason = meta.get("fail_reason") or ""
    dic = meta.get("dic")
    if require_constrained() and not bool(meta.get("constraint_ok")):
        if not fail_reason and meta.get("unconstrained"):
            fail_reason = "unconstrained:" + ",".join(meta["unconstrained"])
        if not fail_reason:
            fail_reason = "constraint_failed"
        return {
            **job,
            "fitness": FAILED_FITNESS,
            "chi2": chi2,
            "k": count_free_params(job["expr"]),
            "likelihood_backend": likelihood_backend,
            "fail_reason": fail_reason,
        }
    fitness, chi2, k = apply_fitness_metric(chi2, job["expr"], dic=dic)
    return {
        **job,
        "fitness": fitness,
        "chi2": chi2,
        "k": k,
        "likelihood_backend": likelihood_backend,
        "fail_reason": fail_reason,
    }


def evaluate_jobs_parallel(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run evaluation jobs; preserves input order."""
    if not jobs:
        return []

    gpu_ids = gpu_ids_from_env()
    for slot, job in enumerate(jobs):
        job["cuda_device"] = cuda_device_for_slot(slot, gpu_ids)

    n_workers = min(parallel_workers(), len(jobs))
    if n_workers <= 1:
        return [_evaluate_job(job) for job in jobs]

    print(
        f"Parallel evaluation: {len(jobs)} models, "
        f"{n_workers} workers, GPUs={gpu_ids or '(all visible)'}",
        flush=True,
    )

    ctx = get_context("spawn")
    ordered: list[dict[str, Any] | None] = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        future_map = {
            pool.submit(_evaluate_job, job): idx for idx, job in enumerate(jobs)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            ordered[idx] = future.result()

    return [r for r in ordered if r is not None]


def make_parallel_cal_pop_fitness(
    ga,
    *,
    cache_get: Callable,
    cache_set: Callable,
    mask_key_fn: Callable,
    enforce_mask_fn: Callable,
    mask_to_chromosome_fn: Callable,
):
    """
    Build a zero-arg ``cal_pop_fitness`` replacement (PyGAD calls
    ``self.cal_pop_fitness()`` with no arguments).

    Batch-evaluates pending individuals. Reads ``_seen_masks_this_gen``
    from ``GA_MCMC_new`` each call (reset per generation).
    """

    def parallel_cal_pop_fitness():
        import GA_MCMC_new as ga_main

        seen_masks = ga_main._seen_masks_this_gen
        import numpy as np

        pop_fitness: list[Any] = ["undefined"] * len(ga.population)

        if ga.last_generation_elitism is not None:
            last_elitism_as_list = ga.last_generation_elitism.tolist()
        else:
            last_elitism_as_list = []

        if ga.last_generation_parents is not None:
            last_parents_as_list = ga.last_generation_parents.tolist()
        else:
            last_parents_as_list = []

        pending: list[dict[str, Any]] = []

        for sol_idx, sol in enumerate(ga.population):
            sol = enforce_mask_fn(sol)
            key = mask_key_fn(sol)

            cached_fitness, _backend = cache_get(key)
            if cached_fitness is not None:
                pop_fitness[sol_idx] = cached_fitness
                continue

            if key in seen_masks:
                # Fitness is chi2 (lower better); failed / skipped → +inf.
                cache_set(key, np.inf, "skipped_duplicate")
                pop_fitness[sol_idx] = np.inf
                continue

            if (
                ga.keep_elitism > 0
                and ga.last_generation_elitism is not None
                and len(ga.last_generation_elitism) > 0
                and list(sol) in last_elitism_as_list
            ):
                elitism_idx = last_elitism_as_list.index(list(sol))
                elitism_idx = ga.last_generation_elitism_indices[elitism_idx]
                pop_fitness[sol_idx] = ga.previous_generation_fitness[elitism_idx]
                continue

            if (
                (ga.keep_parents == -1 or ga.keep_parents > 0)
                and ga.last_generation_parents is not None
                and len(ga.last_generation_parents) > 0
                and list(sol) in last_parents_as_list
            ):
                parent_idx = last_parents_as_list.index(list(sol))
                parent_idx = ga.last_generation_parents_indices[parent_idx]
                pop_fitness[sol_idx] = ga.previous_generation_fitness[parent_idx]
                continue

            seen_masks.add(key)
            chrom = mask_to_chromosome_fn(sol)
            expr = chrom.expr_str
            if "=" in expr:
                expr = expr.split("=", 1)[1].strip()

            gen = ga.generations_completed
            pending.append(
                {
                    "sol_idx": sol_idx,
                    "key": key,
                    "gen": gen,
                    "idx": sol_idx,
                    "expr": expr,
                    "model_name": f"gen_{gen}_model_{sol_idx}",
                    "pair_summary": chrom.pair_summary(),
                }
            )

        if pending:
            for job in pending:
                print(
                    f"\n>>> GA generation {job['gen']}, individual {job['idx']} "
                    f"(queued): {job['expr']}",
                    flush=True,
                )
            results = evaluate_jobs_parallel(pending)
            for res in results:
                cache_set(
                    res["key"],
                    res["fitness"],
                    res["likelihood_backend"],
                    chi2=res.get("chi2"),
                    k=res.get("k"),
                    fail_reason=res.get("fail_reason") or "",
                )
                pop_fitness[res["sol_idx"]] = res["fitness"]

        for sol_idx, fit in enumerate(pop_fitness):
            if fit == "undefined":
                pop_fitness[sol_idx] = np.inf

        return np.array(pop_fitness)

    return parallel_cal_pop_fitness
