"""
GA_MCMC_new.py
==============
Main genetic-algorithm driver for discovering dynamical dark-energy
equations of state ``w(a)``.

Flow
----
1. Build / repair binary masks via ``chromosome`` (unique shapes, 2–5 actives).
2. Evaluate each individual with Cobaya or CLiENT (``GA_LIKELIHOOD``).
3. Score with DIC (default), BIC, AIC, or chi2 (``GA_FITNESS_METRIC``).
4. Fail unconstrained DE coefficients (``posterior_diagnostics``).
5. Evolve with PyGAD; write ``GA_summary.txt`` / ``GA_run.log``.
6. Optionally run ``post_ga`` on the winner (``GA_RUN_POST``).

Entry point for containers: ``docker/run_ga.sh`` → this file.
Population size: ``INITIAL_POP_SIZE = GA_POP_SIZE`` (env ``GA_POP_SIZE``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Serial runs: mpi4py may be installed without a working libmpi (common in clienv).
os.environ.setdefault("COBAYA_NOMPI", "1")
import random
from datetime import datetime
import numpy as np
import pygad
from collections import Counter

from run_model_results import *
from chromosome import (
    init_population,
    mask_to_chromosome,
    mask_key,
    mask_to_bitstring,
    enforce_max_active_pairs,
    N_BASIS,
    N_COEFFS,
    N_SHAPES,
    MIN_ACTIVE_PAIRS,
    MAX_ACTIVE_PAIRS,
    MAX_EXACT_ENUMERATION,
    pygad_on_mutation,
)
from cobaya_runner import run_cobaya_best_chi2
from ga_parallel import (
    evaluate_jobs_parallel,
    gpu_ids_from_env,
    make_parallel_cal_pop_fitness,
    parallel_workers,
)

# Likelihood backend: "cobaya" (full MCMC) or "client" (CLiENT emulator + Cobaya).
# Set GA_LIKELIHOOD=client and CLIENT_ROOT=/path/to/client_public (see CLIENT_SETUP.md).
LIKELIHOOD_BACKEND = os.environ.get("GA_LIKELIHOOD", "cobaya").strip().lower()

# =============================================================================
# Cobaya data config — choose which YAML drives likelihoods / theory / sampler.
# Files live in cobaya_configs/. Examples currently available:
#   "cmb_spa"  — DESI + Pantheon+SH0ES + Planck + ACT + SPT + MUSE
#   "sn_desi"  — Pantheon+SH0ES + DESI DR2 only (fast SN+BAO)
# Or pass a full path to any .yaml. Env override: GA_COBAYA_CONFIG / GA_COBAYA_YAML.
# =============================================================================
COBAYA_CONFIG = os.environ.get("GA_COBAYA_CONFIG", "cmb_spa").strip()


def _apply_cobaya_config():
    """Select Cobaya YAML before any MCMC (also sets env for worker subprocesses)."""
    from cobaya_runner import list_cobaya_configs, set_cobaya_yaml

    available = list_cobaya_configs()
    path = set_cobaya_yaml(COBAYA_CONFIG)
    print(
        f"Cobaya config choice: {COBAYA_CONFIG!r}\n"
        f"  → {path}\n"
        f"  available in cobaya_configs/: {available}",
        flush=True,
    )
    return path


def run_likelihood_best_chi2(expr, model_class_dir, model_name, master_classy_dir=None):
    """
    Returns ``(chi2, likelihood_backend, meta)`` where backend is one of:
    client, cobaya, cobaya_fallback. ``meta`` may include ``run_dir``.
    """
    if LIKELIHOOD_BACKEND == "client":
        from client_runner import run_client_best_chi2

        out = run_client_best_chi2(
            expr,
            model_class_dir,
            model_name,
            master_classy_dir,
        )
        if len(out) == 2:
            return out[0], out[1], {}
        return out[0], out[1], out[2] or {}
    if LIKELIHOOD_BACKEND != "cobaya":
        raise ValueError(
            f"Unknown GA_LIKELIHOOD={LIKELIHOOD_BACKEND!r}; use 'cobaya' or 'client'"
        )
    chi2 = run_cobaya_best_chi2(
        expr, model_class_dir, model_name, master_classy_dir
    )
    return chi2, "cobaya", {}


# --- Project paths ---
THIS_FILE = os.path.abspath(__file__)
SCRIPTS_DIR = os.path.dirname(THIS_FILE)
CLASS_DIR = os.path.abspath(os.environ.get("GA_CLASS_DIR", os.path.dirname(SCRIPTS_DIR)))
PROJECT_ROOT = os.path.abspath(
    os.environ.get("GA_PROJECT_ROOT", os.path.dirname(CLASS_DIR))
)

SUMMARY_FILE = os.path.join(PROJECT_ROOT, "GA_summary.txt")
LOG_FILE = os.path.join(PROJECT_ROOT, "GA_run.log")


def _ensure_client_runtime() -> None:
    """
    CLiENT needs sklearn/TensorFlow (clienv). CLASSy must be built with the same
    Python that runs Cobaya/CLiENT, so relaunch the whole GA when needed.
    """
    if LIKELIHOOD_BACKEND != "client":
        return

    from client_runner import client_python

    target = client_python()
    current = os.path.abspath(sys.executable)
    if os.path.abspath(target) == current:
        return

    print(
        f"Relaunching GA with CLiENT Python:\n"
        f"  was: {current}\n"
        f"  now: {target}",
        flush=True,
    )
    os.execv(target, [target, THIS_FILE, *sys.argv[1:]])


class _Tee:
    """Write stdout/stderr to the terminal and a log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._log_file.write(data)
        self._log_file.flush()

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def fileno(self):
        return self._stream.fileno()

    def isatty(self):
        return self._stream.isatty()


def setup_run_logging(log_path=LOG_FILE):
    """Mirror all terminal output to ``log_path`` (append mode)."""
    Path(log_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a", encoding="utf-8")
    log.write("\n" + "=" * 90 + "\n")
    log.write(f"# GA run started {datetime.now().isoformat(timespec='seconds')}\n")
    log.write("=" * 90 + "\n")
    log.flush()
    sys.stdout = _Tee(sys.__stdout__, log)
    sys.stderr = _Tee(sys.__stderr__, log)
    print(f"Logging to {log_path}", flush=True)
    if LIKELIHOOD_BACKEND == "client":
        from client_runner import client_python, log_client_gpu_status

        print(
            "Likelihood backend: CLiENT "
            f"(profile={os.environ.get('CLIENT_PROFILE', 'ga_fast')}, "
            f"CLIENT_ROOT={os.environ.get('CLIENT_ROOT', '(not set)')}, "
            f"python={client_python()})",
            flush=True,
        )
        log_client_gpu_status(client_python())
    else:
        print("Likelihood backend: Cobaya MCMC", flush=True)
    try:
        from cobaya_runner import COBAYA_YAML_PATH, DATASETS

        print(f"Cobaya YAML: {COBAYA_YAML_PATH}", flush=True)
        print(f"DATASETS = {DATASETS}", flush=True)
    except Exception as exc:
        print(f"Could not import Cobaya YAML / DATASETS: {exc}", flush=True)
    n_workers = parallel_workers()
    gpu_ids = gpu_ids_from_env()
    if n_workers > 1:
        print(
            f"Parallel model evaluation: GA_PARALLEL_WORKERS={n_workers}, "
            f"GA_GPU_IDS={gpu_ids or '(all visible to each worker)'}",
            flush=True,
        )
    return log


# --- GA layout (coeff×shape free combination: N_COEFFS × N_SHAPES genes) ---
# Override catalogue with: export GA_INITIAL_CATALOG=canonical_five
_ALLOWED_INITIAL_CATALOGS = (
    "canonical_five",
    "canonical_plus_enumerate",
    "enumerate",
    "full",
)
INITIAL_CATALOG = os.environ.get(
    "GA_INITIAL_CATALOG", "canonical_plus_enumerate"
).strip().lower()
if INITIAL_CATALOG not in _ALLOWED_INITIAL_CATALOGS:
    raise ValueError(
        f"GA_INITIAL_CATALOG must be one of {_ALLOWED_INITIAL_CATALOGS}, "
        f"got {INITIAL_CATALOG!r}"
    )
INCLUDE_ZERO_MODEL = False
EXHAUSTIVE_GEN0 = False
GA_POP_SIZE = int(os.environ.get("GA_POP_SIZE", "20"))
INITIAL_POP_SIZE = GA_POP_SIZE  # gen 0 must match population size
NUM_GENERATIONS = int(os.environ.get("GA_NUM_GENERATIONS", "40"))
MAX_MASK_STRUCTURES = (2 ** N_BASIS) - (0 if INCLUDE_ZERO_MODEL else 1)

# Strong search (see print_ga_settings() in main for plain-English meanings).
NUM_PARENTS_MATING = int(os.environ.get("GA_NUM_PARENTS_MATING", "12"))
MUTATION_PERCENT_GENES = 50
CROSSOVER_TYPE = "uniform"
CROSSOVER_PROBABILITY = 0.95
PARENT_SELECTION_TYPE = "sss"
K_TOURNAMENT = 2
KEEP_ELITISM = 1
PAIR_COL_WIDTH = 44

# Set in main(); fitness callbacks read these globals.
pop = None
POP_SIZE = 0
initial_population = None
_seen_masks_this_gen: set = set()
# mask_key -> (fitness, likelihood_backend, chi2, k, fail_reason)
_fitness_cache: dict = {}


def _cache_get(key):
    """Return (fitness, likelihood_backend) for a mask key."""
    entry = _fitness_cache.get(key)
    if entry is None:
        return None, "unknown"
    if isinstance(entry, tuple):
        return entry[0], entry[1]
    return float(entry), "cobaya"


def _cache_get_full(key):
    """Return (fitness, backend, chi2, k, fail_reason)."""
    entry = _fitness_cache.get(key)
    if entry is None:
        return None, "unknown", None, None, ""
    if isinstance(entry, tuple) and len(entry) >= 5:
        return entry[0], entry[1], entry[2], entry[3], entry[4] or ""
    if isinstance(entry, tuple) and len(entry) >= 4:
        return entry[0], entry[1], entry[2], entry[3], ""
    if isinstance(entry, tuple) and len(entry) == 2:
        return entry[0], entry[1], None, None, ""
    return float(entry), "cobaya", None, None, ""


def _cache_set(key, fitness, likelihood_backend, chi2=None, k=None, fail_reason=""):
    _fitness_cache[key] = (
        fitness,
        likelihood_backend,
        chi2,
        k,
        fail_reason or "",
    )


# Fitness = DIC / BIC / AIC / chi2 (lower better). Failed / skipped → +inf.
FAILED_FITNESS = np.inf


def normalize_fitness_value(raw_fitness):
    """Coerce raw evaluator output to a finite chi2 (not yet BIC/AIC)."""
    if raw_fitness is None:
        return FAILED_FITNESS

    if isinstance(raw_fitness, (int, float, np.integer, np.floating)):
        return float(raw_fitness)

    if isinstance(raw_fitness, tuple) and len(raw_fitness) == 2:
        _, sampler = raw_fitness
        try:
            from cobaya_runner import best_chi2_from_sample_data

            products = sampler.products(to_getdist=False)
            sample = products["sample"]
            return float(best_chi2_from_sample_data(sample.data))
        except Exception as e:
            print(f"[normalize_fitness_value ERROR] {type(e).__name__}: {e}")
            return FAILED_FITNESS

    return FAILED_FITNESS


def apply_fitness_metric(chi2, expr, *, dic=None):
    """
    Map χ² / DIC → GA fitness (DIC by default).

    Returns ``(fitness, chi2, k)``.
    """
    from fitness_metrics import fitness_metric, score_from_chi2, score_from_dic

    if fitness_metric() == "dic" and dic is not None:
        return score_from_dic(dic, chi2, expr)
    return score_from_chi2(chi2, expr)


def is_failed_fitness(fitness):
    """True if fitness is non-finite (failed / skipped model)."""
    return fitness is None or not np.isfinite(fitness)


# --- 2. Likelihood evaluation bridge ---
def evaluate_model_likelihood(expr, model_name):
    """
    Returns ``(chi2, likelihood_backend, meta)``.

    ``meta`` may include ``run_dir``, ``dic``, ``fail_reason``, ``unconstrained``.
    """
    class_dir = None
    try:
        class_dir = prepare_model_directory(model_name)

        generator = UserEquationGenerator(class_dir, model_name, expr=expr)
        generator.run()

        update_class_for_new_params(class_dir, expr)
        compile_class(class_dir, model_name)
        build_model_classy(class_dir, model_name)

        chi2, likelihood_backend, meta = run_likelihood_best_chi2(
            expr,
            class_dir,
            model_name,
            master_classy_dir=None,
        )
        meta = dict(meta or {})

        from posterior_diagnostics import diagnose_client_run, newest_client_run

        run_dir = meta.get("run_dir") or newest_client_run(class_dir)
        model_root = os.path.dirname(class_dir)
        diag = diagnose_client_run(run_dir, expr, model_root=model_root)
        meta.update(
            {
                "run_dir": str(run_dir) if run_dir else None,
                "dic": diag.get("dic"),
                "Dbar": diag.get("Dbar"),
                "p_D": diag.get("p_D"),
                "unconstrained": diag.get("unconstrained") or [],
                "fail_reason": diag.get("fail_reason") or "",
                "constraint_ok": bool(diag.get("ok")),
            }
        )

        print(
            f"Model: {model_name} | Best chi2: {chi2:.4f} | "
            f"DIC={meta.get('dic')} | "
            f"likelihood_backend: {likelihood_backend}"
            + (
                f" | FAIL {meta['fail_reason']}"
                if meta.get("fail_reason")
                else ""
            ),
            flush=True,
        )

        return float(chi2), likelihood_backend, meta

    except Exception as e:
        from cobaya_runner import best_chi2_from_chain

        recovered = (
            best_chi2_from_chain(class_dir, model_name) if class_dir else None
        )
        if recovered is not None:
            print(
                f"[{model_name}] Recovered best chi2={recovered:.4f} "
                f"after {type(e).__name__} | likelihood_backend: cobaya_recovered",
                flush=True,
            )
            return float(recovered), "cobaya_recovered", {
                "fail_reason": "recovered_no_constraint_check",
                "dic": None,
                "constraint_ok": False,
            }
        import traceback
        print(f"\n[ERROR] in model '{model_name}': {type(e).__name__}: {e}")
        traceback.print_exc()
        return FAILED_FITNESS, "failed", {
            "fail_reason": f"error:{type(e).__name__}",
            "constraint_ok": False,
        }


# --- 3. GA fitness function ---
def fitness_func(ga, solution, idx):
    global _seen_masks_this_gen, _fitness_cache

    solution = enforce_max_active_pairs(solution)
    key = mask_key(solution)
    if key in _fitness_cache:
        fitness, _backend = _cache_get(key)
        return fitness

    if key in _seen_masks_this_gen:
        print(
            f"⚠️ Duplicate mask {list(key)} in generation "
            f"{ga.generations_completed} — skipping MCMC",
            flush=True,
        )
        _cache_set(key, FAILED_FITNESS, "skipped_duplicate")
        return FAILED_FITNESS

    _seen_masks_this_gen.add(key)

    chrom = mask_to_chromosome(solution)
    expr = chrom.expr_str

    if "=" in expr:
        expr = expr.split("=", 1)[1].strip()

    # PyGAD sets generations_completed before each cal_pop_fitness() call.
    gen = ga.generations_completed
    model_id = f"gen_{gen}_model_{idx}"
    print(f"\n>>> GA generation {gen}, individual {idx}: {expr}", flush=True)

    raw_chi2, likelihood_backend, meta = evaluate_model_likelihood(expr, model_id)
    chi2 = normalize_fitness_value(raw_chi2)
    meta = meta or {}
    fail_reason = meta.get("fail_reason") or ""
    unconstrained = meta.get("unconstrained") or []
    dic = meta.get("dic")
    constraint_ok = bool(meta.get("constraint_ok"))

    from posterior_diagnostics import require_constrained

    if require_constrained() and not constraint_ok:
        # Hard fail: unconstrained DE coeff (e.g. flat w6) or missing posterior.
        if not fail_reason and unconstrained:
            fail_reason = "unconstrained:" + ",".join(unconstrained)
        if not fail_reason:
            fail_reason = "constraint_failed"
        fitness = FAILED_FITNESS
        from fitness_metrics import count_free_params

        k = count_free_params(expr)
        print(
            f"Model: {model_id} | FAILED constraint | {fail_reason} | "
            f"chi2={chi2:.4f} | k={k}",
            flush=True,
        )
        _cache_set(
            key,
            fitness,
            likelihood_backend,
            chi2=chi2,
            k=k,
            fail_reason=fail_reason,
        )
        return fitness

    fitness, chi2, k = apply_fitness_metric(chi2, expr, dic=dic)
    print(
        f"Model: {model_id} | chi2={chi2:.4f} | k={k} | fitness={fitness:.4f}"
        + (f" | reason={fail_reason}" if fail_reason else ""),
        flush=True,
    )
    _cache_set(
        key,
        fitness,
        likelihood_backend,
        chi2=chi2,
        k=k,
        fail_reason=fail_reason,
    )
    return fitness


# --- 4. Generation callback ---


def write_generation_summary(ga, gen, fitness_values=None):
    """Print and append one generation block to GA_summary.txt."""
    from fitness_metrics import fitness_metric, metric_label

    fitness_list = (
        fitness_values if fitness_values is not None else ga.last_generation_fitness
    )
    metric = fitness_metric()

    print(f"\nGeneration {gen} complete  [fitness = {metric_label()}]")
    print(
        f"{'#':<5} {'pairs':<{PAIR_COL_WIDTH}} {'n_params':<10} {'k':<6} "
        f"{'chi2':<10} {metric:<10} {'likelihood_backend':<18} expr"
    )
    print("-" * 120)

    rows = []
    for i, solution in enumerate(ga.population):
        chrom = mask_to_chromosome(solution)
        fitness = float(fitness_list[i])
        status = "failed" if is_failed_fitness(fitness) else "success"
        _fit, likelihood_backend, chi2_c, k_c, fail_reason = _cache_get_full(
            mask_key(solution)
        )
        if chi2_c is None or (isinstance(chi2_c, float) and not np.isfinite(chi2_c)):
            chi2_c = fitness if metric == "chi2" else np.inf
        if k_c is None:
            k_c = int(chrom.n_params)
        chi2 = float(chi2_c) if np.isfinite(chi2_c) else np.inf
        k_i = int(k_c)
        reason = fail_reason or "-"

        print(
            f"{i:<5} {chrom.pair_summary():<{PAIR_COL_WIDTH}} {chrom.n_params:<10} "
            f"{k_i:<6} {chi2:<10.2f} {fitness:<10.2f} "
            f"{likelihood_backend:<18} {reason:<28} {chrom.expr_str}"
        )
        rows.append(
            (
                f"gen_{gen}_model_{i}",
                chi2,
                fitness,
                status,
                likelihood_backend,
                chrom.expr_str,
                k_i,
                mask_to_bitstring(chrom.mask),
                reason,
            )
        )

    with open(SUMMARY_FILE, "a") as f:
        f.write("\n" + "=" * 90 + "\n")
        f.write(f"# Generation {gen}  (fitness={metric_label()})\n")
        f.write(
            "# Fail rule: if any free DE coeff is unconstrained "
            "(95% CI covers >= GA_UNCONSTRAINED_FRAC of prior width), "
            "status=failed and fail_reason=unconstrained:<params>\n"
        )
        f.write("=" * 90 + "\n")
        f.write(
            f"{'ID':<15} {'chi2':<12} {'fitness':<12} {'k':<6} {'status':<10} "
            f"{'likelihood_backend':<18} {'fail_reason':<28} "
            f"{'mask':<{N_BASIS + 2}} expr\n"
        )
        f.write("-" * 90 + "\n")

        for (
            model_id,
            chi2,
            fitness,
            status,
            likelihood_backend,
            expr,
            k,
            mask_bits,
            reason,
        ) in rows:
            chi2_str = f"{chi2:.4f}" if np.isfinite(chi2) else "inf"
            fitness_str = f"{fitness:.4f}" if np.isfinite(fitness) else "inf"
            f.write(
                f"{model_id:<15} {chi2_str:<12} {fitness_str:<12} "
                f"{k:<6} {status:<10} {likelihood_backend:<18} "
                f"{reason:<28} "
                f"{mask_bits:<{N_BASIS + 2}} {expr}\n"
            )

    backend_counts = Counter(r[4] for r in rows)
    print(f"Likelihood backends: {dict(backend_counts)}", flush=True)

    n_unique = len({r[5] for r in rows})
    print(f"Unique expressions: {n_unique} / {len(rows)}")
    if n_unique < len(rows):
        print("⚠️ Duplicate expression(s) in this generation")
    n_seen = len(_fitness_cache)
    print(
        f"Structures evaluated so far: {n_seen} / {MAX_MASK_STRUCTURES} "
        f"({MAX_MASK_STRUCTURES - n_seen} still unseen)",
        flush=True,
    )

    valid = [r for r in rows if not is_failed_fitness(r[2])]
    if valid:
        # Rank by fitness (BIC/AIC/chi2), not raw chi2
        best = min(valid, key=lambda r: r[2])
        print(
            f"\nBest: {metric}={best[2]:.4f}, chi2={best[1]:.4f}, k={best[6]}, "
            f"backend={best[4]}, fail_reason={best[8]}, mask={best[7]}, "
            f"expr={best[5]}"
        )


def on_generation(ga):
    global _seen_masks_this_gen
    write_generation_summary(ga, ga.generations_completed)
    _seen_masks_this_gen = set()


def _population_snapshot(population_masks, fitness_values):
    """Minimal object for ``write_generation_summary``."""
    snap = type("PopSnap", (), {})()
    snap.population = np.array(population_masks, dtype=int)
    snap.last_generation_fitness = fitness_values
    return snap


def run_full_initial_scan(pop):
    """
    Evaluate every mask in ``pop`` (generation 0 exhaustive scan).
    Returns fitness list and seeds the cache so PyGAD does not re-run MCMC.
    """
    global _seen_masks_this_gen, _fitness_cache

    _seen_masks_this_gen = set()
    masks = [ch.mask for ch in pop]
    fitnesses: list[float] = [0.0] * len(pop)

    print(
        f"\n=== Generation 0: exhaustive scan ({len(pop)} models) ===",
        flush=True,
    )

    jobs = []
    for idx, ch in enumerate(pop):
        key = mask_key(ch.mask)
        if key in _fitness_cache:
            fitness, _ = _cache_get(key)
            fitnesses[idx] = fitness
            continue

        _seen_masks_this_gen.add(key)
        expr = ch.expr_str
        if "=" in expr:
            expr = expr.split("=", 1)[1].strip()

        jobs.append(
            {
                "idx": idx,
                "key": key,
                "gen": 0,
                "expr": expr,
                "model_name": f"gen_0_model_{idx}",
            }
        )

    if jobs:
        for job in jobs:
            print(
                f"\n>>> GA generation 0, individual {job['idx']} "
                f"(queued): {job['expr']}",
                flush=True,
            )
        results = evaluate_jobs_parallel(jobs)
        for res in results:
            _cache_set(
                res["key"],
                res["fitness"],
                res["likelihood_backend"],
                chi2=res.get("chi2"),
                k=res.get("k"),
                fail_reason=res.get("fail_reason") or "",
            )
            fitnesses[res["idx"]] = res["fitness"]

    snap = _population_snapshot(masks, fitnesses)
    write_generation_summary(snap, gen=0, fitness_values=fitnesses)
    _seen_masks_this_gen = set()
    return fitnesses


def select_seed_population(pop, fitnesses, size):
    """Top-``size`` masks by fitness (lowest BIC/AIC/chi2) to start the GA."""
    from fitness_metrics import fitness_metric, metric_label

    ranked = sorted(
        range(len(pop)),
        key=lambda i: fitnesses[i],
    )
    chosen = ranked[:size]
    metric = fitness_metric()
    print(
        f"\nSeeding generations 1–{NUM_GENERATIONS} with top {size} models "
        f"(lowest {metric_label()}) from generation 0:",
        flush=True,
    )
    for rank, i in enumerate(chosen, 1):
        ch = pop[i]
        fit = fitnesses[i] if np.isfinite(fitnesses[i]) else np.inf
        fit_str = f"{fit:.4f}" if np.isfinite(fit) else "inf"
        _f, _b, chi2_c, k_c, _reason = _cache_get_full(mask_key(ch.mask))
        extra = ""
        if chi2_c is not None and np.isfinite(chi2_c):
            extra = f"  chi2={chi2_c:.4f}"
        if k_c is not None:
            extra += f"  k={k_c}"
        print(
            f"  {rank:>2}. {ch.pair_summary():<35} {metric}={fit_str}{extra}  "
            f"{ch.expr_str}"
        )
    return [enforce_max_active_pairs(pop[i].mask) for i in chosen]


def parent_selection_minimize(fitness, num_parents, ga_instance):
    """
    Parent selection for chi2 fitness (lower better).

    PyGAD maximizes by default; this callable replaces built-in selection so
    parents are the lowest-chi2 individuals (sss) or tournament winners (min).
    """
    fitness = np.asarray(fitness, dtype=float)
    n = len(fitness)
    num_parents = int(num_parents)

    if PARENT_SELECTION_TYPE == "tournament":
        k = max(2, int(getattr(ga_instance, "K_tournament", K_TOURNAMENT)))
        parents_indices = []
        for _ in range(num_parents):
            cand = np.random.randint(0, n, size=k)
            winner = cand[int(np.argmin(fitness[cand]))]
            parents_indices.append(int(winner))
        parents_indices = np.array(parents_indices, dtype=int)
    else:
        # Steady-state style: take the num_parents lowest chi2 scores.
        parents_indices = np.argsort(fitness)[:num_parents]

    parents = ga_instance.population[parents_indices, :].copy()
    return parents, parents_indices


def make_steady_state_selection_minimize(ga):
    """
    Replace PyGAD's maximize SSS used for elitism / reserved elites so the
    lowest-chi2 individuals are kept.
    """

    def steady_state_selection_minimize(fitness, num_parents):
        fitness = np.asarray(fitness, dtype=float)
        indices = np.argsort(fitness)[: int(num_parents)]
        parents = ga.population[indices, :].copy()
        return parents, indices

    return steady_state_selection_minimize


def best_solution_minimize(ga, pop_fitness=None):
    """Best individual = lowest chi2 (PyGAD's best_solution maximizes)."""
    if pop_fitness is None:
        pop_fitness = ga.last_generation_fitness
    pop_fitness = np.asarray(pop_fitness, dtype=float)
    best_idx = int(np.argmin(pop_fitness))
    return ga.population[best_idx].copy(), float(pop_fitness[best_idx]), best_idx


def print_ga_settings(num_parents: int):
    """Summarise GA operator choices for the log / terminal."""
    from math import comb

    from fitness_metrics import metric_label
    from posterior_diagnostics import require_constrained, unconstrained_frac

    n_valid = sum(
        comb(N_SHAPES - 1, k) for k in range(1, MAX_ACTIVE_PAIRS)
    )
    print(
        "\n--- GA operator settings ---\n"
        f"  Fitness                 : {metric_label()} "
        "(lower better; failed → +inf)\n"
        f"  Constraint gate         : "
        f"{'ON' if require_constrained() else 'OFF'} "
        f"(fail if any DE coeff 95% CI covers ≥{unconstrained_frac():.0%} "
        "of prior; env GA_REQUIRE_CONSTRAINED / GA_UNCONSTRAINED_FRAC)\n"
        f"  Encoding                : {N_COEFFS} coeffs × {N_SHAPES} shapes "
        f"= {N_BASIS} genes (free pairs; repair → unique shapes)\n"
        f"  Active shapes           : {MIN_ACTIVE_PAIRS}–{MAX_ACTIVE_PAIRS} "
        f"(const always on) → ~{n_valid:,} distinct models\n"
        f"  Search space (raw genes): 2^{N_BASIS} ≈ {MAX_MASK_STRUCTURES:,} masks\n"
        f"  Initial catalogue       : {INITIAL_CATALOG!r} "
        f"(env GA_INITIAL_CATALOG)\n"
        f"  Population              : {GA_POP_SIZE} each generation "
        f"(env GA_POP_SIZE)\n"
        f"  Generations             : {NUM_GENERATIONS} "
        f"(env GA_NUM_GENERATIONS)\n"
        f"  Parents per mating pool : {num_parents}  (lowest-fitness parents)\n"
        f"  Crossover               : {CROSSOVER_TYPE!r} @ {CROSSOVER_PROBABILITY:.0%}  "
        "(each gene randomly from parent A or B)\n"
        f"  Mutation                : {MUTATION_PERCENT_GENES}% of genes flipped per offspring\n"
        f"  Parent selection        : {PARENT_SELECTION_TYPE!r} (minimize fitness)\n"
        f"  Elitism                 : keep lowest-fitness {KEEP_ELITISM} individual(s)\n"
        "---\n",
        flush=True,
    )


def run_ga(ga, log_gen0_in_pygad=True):
    """Run PyGAD; optionally log generation 0 inside PyGAD (skipped after full scan)."""
    if parallel_workers() > 1:
        ga.cal_pop_fitness = make_parallel_cal_pop_fitness(
            ga,
            cache_get=_cache_get,
            cache_set=_cache_set,
            mask_key_fn=mask_key,
            enforce_mask_fn=enforce_max_active_pairs,
            mask_to_chromosome_fn=mask_to_chromosome,
        )

    if not log_gen0_in_pygad:
        ga._gen0_logged = True
        original_cal = ga.cal_pop_fitness
        ga.cal_pop_fitness = original_cal
        ga.run()
        return

    original_cal = ga.cal_pop_fitness
    ga._gen0_logged = False

    def cal_pop_fitness_with_gen0_log():
        fitness = original_cal()
        if not ga._gen0_logged and ga.generations_completed == 0:
            ga._gen0_logged = True
            write_generation_summary(ga, gen=0, fitness_values=fitness)
        return fitness

    ga.cal_pop_fitness = cal_pop_fitness_with_gen0_log
    ga.run()


def main():
    global pop, POP_SIZE, initial_population, _seen_masks_this_gen, _fitness_cache

    _fitness_cache = {}

    if EXHAUSTIVE_GEN0 and INITIAL_CATALOG != "full":
        raise ValueError("EXHAUSTIVE_GEN0=True requires INITIAL_CATALOG='full'")
    if EXHAUSTIVE_GEN0 and N_BASIS > MAX_EXACT_ENUMERATION:
        raise ValueError(
            f"EXHAUSTIVE_GEN0 not supported for N_BASIS={N_BASIS} "
            f"(>{MAX_EXACT_ENUMERATION}); use partial gen 0"
        )

    gen0_size = MAX_MASK_STRUCTURES if EXHAUSTIVE_GEN0 else INITIAL_POP_SIZE
    pop = init_population(
        output_path=os.path.join(SCRIPTS_DIR, "user_eq.txt"),
        size=gen0_size,
        initial_catalog=INITIAL_CATALOG,
        include_zero_model=INCLUDE_ZERO_MODEL,
        seed=2026,
    )
    n_gen0 = len(pop)
    if EXHAUSTIVE_GEN0 and n_gen0 != MAX_MASK_STRUCTURES:
        raise ValueError(
            f"exhaustive gen 0 expected {MAX_MASK_STRUCTURES} models, got {n_gen0}"
        )
    if not EXHAUSTIVE_GEN0 and n_gen0 != GA_POP_SIZE:
        raise ValueError(
            f"generation 0 size ({n_gen0}) must equal GA_POP_SIZE ({GA_POP_SIZE})"
        )

    mode = "exhaustive" if EXHAUSTIVE_GEN0 else "partial (discovery)"
    print(
        f"Generation 0: {INITIAL_CATALOG!r}, {n_gen0} models [{mode}]\n"
        f"Encoding: {N_COEFFS} coeffs × {N_SHAPES} shapes = {N_BASIS} genes\n"
        f"Search space (theoretical): 2^{N_BASIS} ≈ {MAX_MASK_STRUCTURES:,} masks\n",
        flush=True,
    )
    print(f"{'#':<5} {'pairs':<{PAIR_COL_WIDTH}} {'n_params':<10} expr")
    print("-" * 90)

    for i, ch in enumerate(pop):
        print(f"{i:<5} {ch.pair_summary():<{PAIR_COL_WIDTH}} {ch.n_params:<10} {ch.expr_str}")

    mask_counts = Counter(tuple(ch.mask) for ch in pop)
    print(f"\nUnique structures: {len(mask_counts)} / {n_gen0}")

    gen0_fitness = run_full_initial_scan(pop)
    if EXHAUSTIVE_GEN0:
        initial_population = select_seed_population(pop, gen0_fitness, GA_POP_SIZE)
    else:
        initial_population = [enforce_max_active_pairs(ch.mask) for ch in pop]
        print(
            f"\nGenerations 1–{NUM_GENERATIONS}: up to "
            f"{MAX_MASK_STRUCTURES - n_gen0} unevaluated masks remain for discovery",
            flush=True,
        )
    POP_SIZE = GA_POP_SIZE

    num_parents = min(NUM_PARENTS_MATING, POP_SIZE)
    if num_parents < 2:
        num_parents = 2

    print_ga_settings(num_parents)

    ga = pygad.GA(
        num_generations=NUM_GENERATIONS,
        sol_per_pop=GA_POP_SIZE,
        num_parents_mating=num_parents,
        num_genes=N_BASIS,
        gene_space=[[0, 1]] * N_BASIS,
        gene_type=int,
        initial_population=initial_population,
        fitness_func=fitness_func,
        parent_selection_type=parent_selection_minimize,
        K_tournament=K_TOURNAMENT,
        crossover_type=CROSSOVER_TYPE,
        crossover_probability=CROSSOVER_PROBABILITY,
        mutation_type="random",
        mutation_percent_genes=MUTATION_PERCENT_GENES,
        mutation_by_replacement=True,
        keep_elitism=KEEP_ELITISM,
        on_mutation=pygad_on_mutation,
        on_generation=on_generation,
        random_seed=42,
        suppress_warnings=True,
    )
    # Elitism / reserved elites also go through steady_state_selection (maximize
    # by default) — patch to keep lowest chi2 instead.
    ga.steady_state_selection = make_steady_state_selection_minimize(ga)

    print(
        f"Evolving {NUM_GENERATIONS} generations @ pop_size={GA_POP_SIZE}",
        flush=True,
    )
    _seen_masks_this_gen = set()
    run_ga(ga, log_gen0_in_pygad=False)

    solution, fitness, idx = best_solution_minimize(ga)
    best_chrom = mask_to_chromosome(solution)
    _f, _b, best_chi2, best_k, _best_reason = _cache_get_full(mask_key(solution))
    from fitness_metrics import fitness_metric, metric_label

    print("\n=== FINAL RESULT ===")
    print(f"Best pairs: {best_chrom.pair_summary()}")
    print(f"Best expr:  {best_chrom.expr_str}")
    print(f"Fitness ({metric_label()}): {fitness:.4f}")
    if best_chi2 is not None and np.isfinite(best_chi2):
        print(f"chi2: {best_chi2:.4f}  k={best_k}")

    # Optional post-GA CLiENT (T=1) + GetDist — see post_ga.py.
    # Disable with: export GA_RUN_POST=0
    # Profile: GA_POST_PROFILE=ga_final_hpc (HPC) or ga_final (default / laptop)
    if os.environ.get("GA_RUN_POST", "1").strip().lower() not in ("0", "false", "no"):
        print("\n=== POST-GA (CLiENT + plots) ===", flush=True)
        from post_ga import best_from_ga_summary, run_post_ga

        post_profile = (
            os.environ.get("GA_POST_PROFILE", "").strip()
            or os.environ.get("CLIENT_PROFILE_POST", "").strip()
            or "ga_final"
        )
        print(f"post-GA profile: {post_profile}", flush=True)

        # Winner = lowest fitness (BIC by default) in GA_summary.
        try:
            model_id, expr, ga_chi2 = best_from_ga_summary(Path(SUMMARY_FILE))
        except SystemExit:
            expr = best_chrom.expr_str
            if "=" in expr:
                expr = expr.split("=", 1)[1].strip()
            model_id = "post_ga_best"
            ga_chi2 = (
                float(best_chi2)
                if best_chi2 is not None and np.isfinite(best_chi2)
                else (float(fitness) if np.isfinite(fitness) else None)
            )
            print(
                f"WARNING: could not read {SUMMARY_FILE}; "
                f"using final population best as {model_id!r}",
                flush=True,
            )

        run_post_ga(
            expr=expr,
            model_name=model_id,
            profile=post_profile,
            cobaya_config=COBAYA_CONFIG,
            ga_chi2=ga_chi2,
            rebuild=True,
            plot=True,
        )
    else:
        print(
            "\nSkipping post-GA (GA_RUN_POST=0). "
            "Run later: python post_ga.py   # auto-picks best from GA_summary.txt",
            flush=True,
        )


if __name__ == "__main__":
    _ensure_client_runtime()
    _apply_cobaya_config()
    _log_handle = setup_run_logging()
    try:
        main()
    except Exception:
        # Restore stdio before printing traceback (Tee may hold a closing log).
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        raise
    finally:
        try:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"\nLog saved to {LOG_FILE}", flush=True)
        finally:
            _log_handle.close()