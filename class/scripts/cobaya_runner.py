import os

os.environ.setdefault("COBAYA_NOMPI", "1")

from cobaya.run import run
from cobaya.log import logger_setup
import json
import logging
import subprocess
import sys
import tempfile
import traceback
from copy import deepcopy

import numpy as np
import yaml

from params import build_ga_params
from classy_path import prepend_model_classy_path

__all__ = [
    "DATASETS",
    "COBAYA_YAML_PATH",
    "clear_cached_classy_modules",
    "get_base_info",
    "list_cobaya_configs",
    "load_cobaya_template",
    "resolve_cobaya_yaml",
    "set_cobaya_yaml",
    "write_cobaya_yaml",
    "refresh_existing_cobaya_yaml",
    "chi2_sn_at_params",
    "chi2_at_params",
    "best_chi2_from_sample_data",
    "cobaya_likelihood_dict",
    "run_cobaya",
    "run_cobaya_best_chi2",
    "prepend_model_classy_path",
    "build_info_params",
]

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIGS_DIR = os.path.join(_SCRIPTS_DIR, "cobaya_configs")
_DEFAULT_CONFIG_NAME = "cmb_spa"

# Active Cobaya YAML path (set by set_cobaya_yaml / resolve at import).
COBAYA_YAML_PATH = None  # filled by set_cobaya_yaml() below

# Optional cap for GA speed: export GA_COBAYA_MAX_SAMPLES=500
# If unset, the YAML sampler block is used as-is (full R-1 stop for CMB-SPA).
_GA_MAX_SAMPLES = os.environ.get("GA_COBAYA_MAX_SAMPLES", "").strip()

# Cobaya console: full debug (every CLASS/prior eval) vs MCMC step lines only.
COBAYA_DEBUG = False
COBAYA_MCMC_STEPS = True  # New sample #N / Progress @ ... (without classy spam)

_template_cache = None
DATASETS = []  # refreshed whenever the YAML is (re)selected


def list_cobaya_configs():
    """Return available config stems in ``cobaya_configs/`` (without .yaml)."""
    if not os.path.isdir(_CONFIGS_DIR):
        return []
    names = []
    for fn in sorted(os.listdir(_CONFIGS_DIR)):
        if fn.endswith(".yaml") or fn.endswith(".yml"):
            names.append(os.path.splitext(fn)[0])
    return names


def resolve_cobaya_yaml(name_or_path):
    """
    Resolve a config choice to an absolute YAML path.

    Accepts:
      - stem in cobaya_configs/  e.g. ``"cmb_spa"`` → cobaya_configs/cmb_spa.yaml
      - filename in that folder  e.g. ``"cmb_spa.yaml"``
      - absolute or relative path to any .yaml file
    """
    if not name_or_path or not str(name_or_path).strip():
        raise ValueError("Empty Cobaya config name/path")

    raw = str(name_or_path).strip()

    if os.path.isfile(raw):
        return os.path.abspath(raw)

    # Try under cobaya_configs/
    candidates = [
        os.path.join(_CONFIGS_DIR, raw),
        os.path.join(_CONFIGS_DIR, f"{raw}.yaml"),
        os.path.join(_CONFIGS_DIR, f"{raw}.yml"),
    ]
    if not raw.endswith((".yaml", ".yml")):
        candidates.append(os.path.join(_CONFIGS_DIR, f"{raw}.yaml"))

    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)

    available = list_cobaya_configs()
    raise FileNotFoundError(
        f"Cobaya config {raw!r} not found.\n"
        f"Looked in {_CONFIGS_DIR} and as a file path.\n"
        f"Available configs: {available or '(none)'}\n"
        "Add a YAML under cobaya_configs/ or pass a full path."
    )


def set_cobaya_yaml(name_or_path):
    """
    Select which Cobaya YAML the GA uses (clears template cache).

    Also sets ``GA_COBAYA_YAML`` so Cobaya worker subprocesses see the same file.
    """
    global COBAYA_YAML_PATH, _template_cache, DATASETS

    COBAYA_YAML_PATH = resolve_cobaya_yaml(name_or_path)
    os.environ["GA_COBAYA_YAML"] = COBAYA_YAML_PATH
    _template_cache = None
    DATASETS = list(cobaya_likelihood_dict().keys())
    return COBAYA_YAML_PATH


def load_cobaya_template():
    """Load and cache the active Cobaya YAML."""
    global _template_cache, COBAYA_YAML_PATH

    if COBAYA_YAML_PATH is None:
        # Env override, else default config name.
        env = os.environ.get("GA_COBAYA_YAML", "").strip()
        set_cobaya_yaml(env if env else _DEFAULT_CONFIG_NAME)

    if _template_cache is None:
        if not os.path.isfile(COBAYA_YAML_PATH):
            raise FileNotFoundError(
                f"Cobaya YAML not found: {COBAYA_YAML_PATH}\n"
                f"Available: {list_cobaya_configs()}"
            )
        with open(COBAYA_YAML_PATH, encoding="utf-8") as f:
            _template_cache = yaml.safe_load(f)
        if not isinstance(_template_cache, dict):
            raise ValueError(f"Invalid Cobaya YAML (not a mapping): {COBAYA_YAML_PATH}")
        if "likelihood" not in _template_cache:
            raise ValueError(f"Cobaya YAML missing 'likelihood': {COBAYA_YAML_PATH}")
    return deepcopy(_template_cache)


def cobaya_likelihood_dict():
    """Full Cobaya ``likelihood`` block from the YAML (not flattened to null)."""
    like = load_cobaya_template().get("likelihood") or {}
    if not like:
        raise ValueError(f"Empty likelihood block in {COBAYA_YAML_PATH}")
    return deepcopy(like)


# Initialise from GA_COBAYA_YAML or default ``cmb_spa``.
set_cobaya_yaml(os.environ.get("GA_COBAYA_YAML", _DEFAULT_CONFIG_NAME))


def best_chi2_from_sample_data(data):
    """
    Best total chi2 from a Cobaya sample table (all datasets combined).

    Prefers column ``chi2``, else sums ``chi2__*`` likelihood columns,
    else falls back to ``chi2__SN``.
    """
    if data is None or getattr(data, "empty", True):
        raise ValueError("empty Cobaya sample")

    if "chi2" in data.columns:
        finite = data["chi2"].replace([np.inf, -np.inf], np.nan).dropna()
        if not finite.empty:
            return float(finite.min())

    like_cols = [
        c for c in data.columns
        if c.startswith("chi2__") and c not in ("chi2__prior", "chi2__theory")
    ]
    if like_cols:
        total = data[like_cols].sum(axis=1)
        finite = total.replace([np.inf, -np.inf], np.nan).dropna()
        if not finite.empty:
            return float(finite.min())

    if "chi2__SN" in data.columns:
        finite = data["chi2__SN"].replace([np.inf, -np.inf], np.nan).dropna()
        if not finite.empty:
            return float(finite.min())

    raise ValueError(
        f"No chi2 column in sample. Available: {list(data.columns)}"
    )


def clear_cached_classy_modules():
    """Drop prior model's classy from sys.modules before loading another build."""
    import importlib

    for name in list(sys.modules):
        if name == "classy" or name.startswith("classy."):
            del sys.modules[name]
    importlib.invalidate_caches()


# --- MPI check ---
try:
    from cobaya.mpi import rank
except ImportError:
    rank = 0  # fallback for serial runs


def _setup_cobaya_logging(debug=COBAYA_DEBUG, mcmc_steps=COBAYA_MCMC_STEPS):
    """
    Configure Cobaya logging for the GA pipeline.

    - COBAYA_DEBUG=True  → everything (classy, prior, model, …)
    - COBAYA_MCMC_STEPS=True (default) → MCMC only: New sample #N, Progress @ …
    - both False → errors and final messages only
    """
    logger_setup(debug=debug)
    if debug:
        return

    for name in logging.root.manager.loggerDict:
        if name.startswith("cobaya") and name not in ("cobaya.mcmc", "cobaya"):
            logging.getLogger(name).setLevel(logging.WARNING)

    mcmc_log = logging.getLogger("cobaya.mcmc")
    mcmc_log.setLevel(logging.DEBUG if mcmc_steps else logging.INFO)


def chain_prefix(model_class_dir, model_name):
    return os.path.join(model_class_dir, "scripts", "chains", model_name)


def best_chi2_from_chain(model_class_dir, model_name):
    """Load best total chi2 from saved Cobaya chain files (partial runs OK)."""
    from cobaya.output import load_samples

    prefix = chain_prefix(model_class_dir, model_name)
    try:
        samples = load_samples(prefix, combined=True, to_getdist=False)
    except FileNotFoundError:
        return None
    data = getattr(samples, "data", None)
    try:
        return best_chi2_from_sample_data(data)
    except ValueError:
        return None


def build_info_params(expr):
    """Params for one GA model: YAML params + fluid DE + extra w_i."""
    template = load_cobaya_template()
    return build_ga_params(expr, template.get("params"))


def get_base_info(model_class_dir, model_name, master_classy_dir):
    """
    Cobaya info for one GA individual.

    Starts from ``COBAYA_YAML_PATH`` (CMB-SPA by default), then overlays:
    - model-specific ``classy`` path
    - ``fluid_equation_of_state: CLP`` (required for GA w(a))
    - output / resume / force for this model folder
    - optional ``GA_COBAYA_MAX_SAMPLES``
    """
    output_dir = chain_prefix(model_class_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    classy_path = os.path.join(model_class_dir, "python")
    if master_classy_dir and not os.path.isdir(classy_path):
        classy_path = master_classy_dir

    info = load_cobaya_template()
    likelihoods = deepcopy(info["likelihood"])
    print(f"Cobaya YAML: {COBAYA_YAML_PATH}", flush=True)
    print(f"Cobaya datasets: {list(likelihoods)}", flush=True)

    # Theory: keep YAML extra_args (hmcode, N_ur, …), force GA fluid + model classy.
    theory = info.setdefault("theory", {})
    classy = theory.setdefault("classy", {})
    classy["path"] = classy_path
    classy["ignore_obsolete"] = True
    classy.setdefault("stop_at_error", False)
    extra = classy.setdefault("extra_args", {})
    extra["fluid_equation_of_state"] = "CLP"
    # Cobaya rejects having both "non_linear" and "non linear".
    if "non linear" in extra:
        if "non_linear" not in extra:
            extra["non_linear"] = extra["non linear"]
        del extra["non linear"]

    sampler = info.setdefault("sampler", {}).setdefault("mcmc", {})
    sampler.setdefault("output_every", "30s")
    if _GA_MAX_SAMPLES:
        sampler["max_samples"] = int(_GA_MAX_SAMPLES)
        print(f"GA_COBAYA_MAX_SAMPLES={sampler['max_samples']}", flush=True)

    info["output"] = output_dir
    info["resume"] = False
    info["force"] = True
    info["debug"] = COBAYA_DEBUG
    # Params are attached by callers via build_info_params(expr).
    info.pop("params", None)

    return info


def write_cobaya_yaml(expr, model_class_dir, model_name, yaml_path, master_classy_dir=None):
    """Write a Cobaya input YAML for CLiENT or standalone runs."""
    info = get_base_info(model_class_dir, model_name, master_classy_dir)
    info["params"] = build_info_params(expr)
    os.makedirs(os.path.dirname(os.path.abspath(yaml_path)), exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(info, f, sort_keys=False, default_flow_style=False)
    return yaml_path


def refresh_existing_cobaya_yaml(
    expr,
    model_class_dir,
    model_name,
    yaml_path,
    master_classy_dir=None,
):
    """
    Keep likelihoods / sampler from an existing model yaml; only refresh
    ``classy`` path and ``params`` (so unused ``wa_fld`` stays fixed, etc.).
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(yaml_path)

    with open(yaml_path, encoding="utf-8") as f:
        info = yaml.safe_load(f)
    if not isinstance(info, dict) or "likelihood" not in info:
        raise ValueError(f"Not a Cobaya yaml with likelihoods: {yaml_path}")

    classy_path = os.path.join(model_class_dir, "python")
    if master_classy_dir and not os.path.isdir(classy_path):
        classy_path = master_classy_dir

    theory = info.setdefault("theory", {})
    classy = theory.setdefault("classy", {})
    classy["path"] = classy_path
    classy["ignore_obsolete"] = True
    classy.setdefault("stop_at_error", False)
    extra = classy.setdefault("extra_args", {})
    extra["fluid_equation_of_state"] = "CLP"
    if "non linear" in extra:
        if "non_linear" not in extra:
            extra["non_linear"] = extra["non linear"]
        del extra["non linear"]

    output_dir = chain_prefix(model_class_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    info["output"] = output_dir
    info["resume"] = False
    info["force"] = True

    info["params"] = build_ga_params(expr, info.get("params"))

    likes = list((info.get("likelihood") or {}).keys())
    print(f"Cobaya YAML (from model folder): {yaml_path}", flush=True)
    print(f"Cobaya datasets: {likes}", flush=True)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(info, f, sort_keys=False, default_flow_style=False)
    return yaml_path


def chi2_at_params(
    expr,
    model_class_dir,
    model_name,
    param_values,
    master_classy_dir=None,
):
    """Evaluate total data chi2 (all YAML likelihoods) at a fixed parameter point."""
    from cobaya.model import get_model

    info = get_base_info(model_class_dir, model_name, master_classy_dir)
    info["params"] = build_info_params(expr)

    _setup_cobaya_logging(debug=False, mcmc_steps=False)
    prepend_model_classy_path(model_class_dir)
    clear_cached_classy_modules()
    model = get_model(info)

    sampled = list(model.parameterization.sampled_params())
    point = {p: float(param_values[p]) for p in sampled if p in param_values}
    if len(point) != len(sampled):
        missing = [p for p in sampled if p not in point]
        raise ValueError(f"Missing parameters for point evaluation: {missing}")

    loglikes, _derived = model.loglikes(
        point, return_derived=True, as_dict=True
    )
    if isinstance(loglikes, dict):
        total = 0.0
        for name, logl in loglikes.items():
            if logl is None:
                raise ValueError(f"likelihood {name!r} returned None")
            total += float(-2.0 * logl)
        return total

    arr = np.asarray(loglikes, dtype=float)
    return float((-2.0 * arr).sum())


def chi2_sn_at_params(
    expr,
    model_class_dir,
    model_name,
    param_values,
    master_classy_dir=None,
):
    """Alias kept for older callers: total chi2 over YAML likelihoods."""
    return chi2_at_params(
        expr, model_class_dir, model_name, param_values, master_classy_dir
    )


def _run_cobaya_in_process(expr, model_class_dir, model_name, master_classy_dir):
    info = get_base_info(model_class_dir, model_name, master_classy_dir)
    info["params"] = build_info_params(expr)

    _setup_cobaya_logging(COBAYA_DEBUG)
    prepend_model_classy_path(model_class_dir)
    clear_cached_classy_modules()
    updated_info, sampler = run(info)

    if rank == 0:
        logging.getLogger("cobaya").info(
            f"Cobaya run for model '{model_name}' completed successfully."
        )
    return updated_info, sampler


def run_cobaya_best_chi2(expr, model_class_dir, model_name, master_classy_dir=None):
    """
    Run Cobaya in a fresh Python subprocess and return the best total chi2
    over all likelihoods in the active Cobaya YAML.

    Uses cobaya_worker.py so multiprocessing does not re-import GA_MCMC_new.py.
    """
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(scripts_dir, "cobaya_worker.py")
    job = {
        "expr": expr,
        "model_class_dir": model_class_dir,
        "model_name": model_name,
        "master_classy_dir": master_classy_dir,
    }

    with tempfile.TemporaryDirectory(prefix="cobaya_job_") as tmp:
        job_path = os.path.join(tmp, "job.json")
        result_path = os.path.join(tmp, "result.json")
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f)

        print(f"\n=== Cobaya MCMC: {model_name} ===", flush=True)
        print(f"Cobaya YAML: {COBAYA_YAML_PATH}", flush=True)
        print(f"DATASETS = {DATASETS}", flush=True)
        worker_env = os.environ.copy()
        worker_env.setdefault("COBAYA_NOMPI", "1")
        proc = subprocess.run(
            [sys.executable, worker, job_path, result_path],
            cwd=scripts_dir,
            env=worker_env,
        )

        if not os.path.isfile(result_path):
            raise RuntimeError(
                f"Cobaya worker for '{model_name}' failed (exit {proc.returncode}): "
                "no result file (see Cobaya log output above)"
            )

        with open(result_path, encoding="utf-8") as f:
            payload = json.load(f)

    if payload.get("status") != "ok":
        recovered = best_chi2_from_chain(model_class_dir, model_name)
        if recovered is not None:
            warn = payload.get("error", "unknown error")
            print(
                f"[{model_name}] Cobaya stopped early ({warn}); "
                f"using best chi2 from saved chain: {recovered:.4f}",
                flush=True,
            )
            return recovered
        tb = payload.get("traceback", "")
        err = payload.get("error", "unknown error")
        if rank == 0:
            logging.error(f"Cobaya run for model '{model_name}' failed: {err}\n{tb}")
        raise RuntimeError(err)

    if payload.get("warning"):
        print(f"[{model_name}] {payload['warning']}", flush=True)
    return float(payload["chi2"])


# --- Main Cobaya runner ---
def run_cobaya(expr, model_class_dir, model_name, master_classy_dir):
    """
    Run Cobaya for a given model dynamically.

    Parameters
    ----------
    expr : str
        Fluid or model expression.
    model_class_dir : str
        Model-specific CLASS source folder.
    model_name : str
        Name of the model.
    master_classy_dir : str
        Path to the master CLASS Python wrapper.
    """
    try:
        return _run_cobaya_in_process(expr, model_class_dir, model_name, master_classy_dir)
    except Exception as e:
        if rank == 0:
            logging.error(f"Cobaya run for model '{model_name}' failed: {e}")
        raise
