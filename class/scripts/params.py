import re
from copy import deepcopy

from class_modifiers import extract_new_parameters


def _symbol_in_expr(expr: str, name: str) -> bool:
    """True if ``name`` appears as a bare identifier in the GA w(a) string."""
    return bool(re.search(rf"\b{re.escape(name)}\b", expr or ""))


def is_lcdm_expr(expr: str) -> bool:
    """True for fluid ΛCDM sentinels: ``w≡-1`` with no free DE coeffs."""
    s = (expr or "").strip().lower().replace(" ", "")
    if s.startswith("w(a)="):
        s = s[5:]
    return s in ("-1", "__lcdm__", "lcdm")

LCDM_PARAMS = {
    "A_s": {
        "latex": "A_\\mathrm{s}",
        "value": "lambda logA: 1e-10*np.exp(logA)",
    },
    "H0": {
        "latex": "H_0",
        "prior": {"max": 100, "min": 20},
        "proposal": 2,
        "ref": {"dist": "norm", "loc": 67, "scale": 2},
    },
    "Omega_Lambda": 0,
    "Omega_m": {"latex": "\\Omega_\\mathrm{m}"},
    "YHe": {"latex": "Y_\\mathrm{P}"},
    "logA": {
        "drop": True,
        "latex": "\\log(10^{10} A_\\mathrm{s})",
        "prior": {"max": 3.91, "min": 1.61},
        "proposal": 0.001,
        "ref": {"dist": "norm", "loc": 3.05, "scale": 0.001},
    },
    "m_ncdm": {"renames": "mnu", "value": 0.06},
    "n_s": {
        "latex": "n_\\mathrm{s}",
        "prior": {"max": 1.2, "min": 0.8},
        "proposal": 0.002,
        "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004},
    },
    "omega_b": {
        "latex": "\\Omega_\\mathrm{b} h^2",
        "prior": {"max": 0.1, "min": 0.005},
        "proposal": 0.0001,
        "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0001},
    },
    "omega_cdm": {
        "latex": "\\Omega_\\mathrm{c} h^2",
        "prior": {"max": 0.99, "min": 0.001},
        "proposal": 0.0005,
        "ref": {"dist": "norm", "loc": 0.12, "scale": 0.001},
    },
    "omegamh2": {
        "derived": "lambda Omega_m, H0: Omega_m*(H0/100)**2",
        "latex": "\\Omega_\\mathrm{m} h^2",
    },
    "tau_reio": {
        "latex": "\\tau_\\mathrm{reio}",
        "prior": {"max": 0.8, "min": 0.01},
        "proposal": 0.003,
        "ref": {"dist": "norm", "loc": 0.055, "scale": 0.006},
    },
    "w0_fld": {
        "latex": "w_{0,\\mathrm{DE}}",
        "prior": {"max": 1, "min": -1.2},
        "proposal": 0.02,
        "ref": {"dist": "norm", "loc": -0.99, "scale": 0.02},
    },
    "wa_fld": {
        "latex": "w_{a,\\mathrm{DE}}",
        "prior": {"max": 2, "min": -3},
        "proposal": 0.05,
        "ref": {"dist": "norm", "loc": 0, "scale": 0.05},
    },
    "z_reio": {"latex": "z_\\mathrm{re}"},
}


# Required for CLASS fluid DE (CLP): cosmological constant off; w(a) is the fluid.
DE_FLUID_PARAMS = {
    "Omega_Lambda": 0,
    "w0_fld": deepcopy(LCDM_PARAMS["w0_fld"]),
    "wa_fld": deepcopy(LCDM_PARAMS["wa_fld"]),
}


def extend_params(expr, base_params):
    """
    Extend the LCDM base parameters with extra parameters
    extracted from the given model expression.
    """
    params = deepcopy(base_params)
    extra_params = extract_new_parameters(expr)

    for p in extra_params:
        if p not in params:
            params[p] = {
                "prior": {"min": -3, "max": 3},
                "proposal": 0.05,
                "ref": {"dist": "norm", "loc": 0.0, "scale": 0.1},
            }
    return params


def build_ga_params(expr, template_params=None):
    """
    Build Cobaya ``params`` for one GA individual.

    Starts from the CMB-SPA (or other) YAML ``params`` block when provided,
    forces fluid-DE settings (``Omega_Lambda=0``, ``w0_fld``, ``wa_fld``),
    then adds any extra ``w_i`` symbols from ``expr``.

    CLASS macros always take ``(a, w0, wa, ...)``, but coefficients that do not
    appear in ``expr`` are **fixed to 0** (not sampled). That matches the GA
    “gene off” convention and avoids dead MCMC dimensions (e.g. ``wa`` when the
    mask never turns on a ``wa`` term).
    """
    if template_params is None:
        base = deepcopy(LCDM_PARAMS)
    else:
        base = deepcopy(template_params)
        for key, value in DE_FLUID_PARAMS.items():
            base[key] = deepcopy(value)

    # Fluid ΛCDM: hardcode w≡−1 in CLASS; no free DE coefficients.
    if is_lcdm_expr(expr):
        base["Omega_Lambda"] = 0
        base["w0_fld"] = -1.0
        base["wa_fld"] = 0.0
        return base

    # Unused fluid coeffs are fixed to 0 (GA "gene off"); they do not enter
    # FLUID_W_FLD when absent from expr. CLASS still requires the fields.
    if _symbol_in_expr(expr, "w0"):
        base["w0_fld"] = deepcopy(LCDM_PARAMS["w0_fld"])
    else:
        base["w0_fld"] = 0.0

    if _symbol_in_expr(expr, "wa"):
        base["wa_fld"] = deepcopy(LCDM_PARAMS["wa_fld"])
    else:
        base["wa_fld"] = 0.0

    return extend_params(expr, base)


def build_lcdm_params(template_params=None):
    """Cobaya ``params`` for fluid ΛCDM (``w≡-1``, no free ``w_i``)."""
    return build_ga_params("-1", template_params=template_params)
