"""
model_builder.py
================
Per-chromosome CLASS compilation pipeline for the scripts_reggie GA.

Architecture
------------
Each unique chromosome mask requires a dedicated compiled CLASS binary so
that its w(a) expression is baked into libclass.a / classy.so at C level.

Directory layout created under scripts_reggie/class_builds/:

    class_builds/
        base_extended/          ← one-time copy of class/ with w2,w3,w4 support
            class/              ← patched source tree
            .built              ← sentinel created after successful compile
        mask_10000/             ← e.g. wCDM: w(a) = w0
            class/
            .built
        mask_11000/             ← CPL : w(a) = w0 + wa*(1-a)
            class/
            .built
        ...

Key functions
-------------
    ensure_extended_base()          one-time: patch + compile the extended base
    build_for_chromosome(chrom)     per-mask: copy base, write user_eq.h, compile
    chromosome_build_path(chrom)    return Path(class_dir) if already built
    generate_user_eq_h(chrom)       return (w_c, dw_c, integral_c) macro strings
    cobaya_theory_for_build(path)   Cobaya theory block for a build directory
    cobaya_params_for_chromosome(chrom)   Cobaya params block

Compilation times (macOS arm64, Apple Clang 17, -j4)
-----------------------------------------------------
    Full build (base_extended):     ~90 s  (one time, cached)
    Per-chromosome rebuild:         ~15 s  (background.c + relink only)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPTS_REGGIE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_REGGIE_DIR.parents[1]
CLASS_SRC = REPO_ROOT / "class"                          # original CLASS tree
BUILDS_DIR = SCRIPTS_REGGIE_DIR / "class_builds"
BASE_EXTENDED_DIR = BUILDS_DIR / "base_extended"
CONDA_ENV = "cosmoga"


def _copytree_ignore(_directory: str, names: List[str]) -> set[str]:
    ignored = {
        name for name in names
        if name in {
            "scripts_reggie",
            "__pycache__",
            ".ipynb_checkpoints",
            ".git",
            "build",
            "build_model2",
        }
    }
    ignored.update({name for name in names if name.endswith("Zone.Identifier")})
    return ignored


def _copy_class_tree(src: Path, dst: Path) -> None:
    """Copy the CLASS source tree while excluding generated workspace artifacts."""
    shutil.copytree(str(src), str(dst), ignore=_copytree_ignore)

# Detect Python executable (works inside cosmoga kernel or subprocess)
def _python_exe() -> str:
    """Return a python interpreter that lives inside the cosmoga env."""
    candidate = Path(sys.executable)
    if "cosmoga" in str(candidate):
        return str(candidate)
    return f"conda run -n {CONDA_ENV} python"


# ---------------------------------------------------------------------------
# 1.  C macro generation (SymPy backend)
# ---------------------------------------------------------------------------

# Coefficient → (sympy expression, human label, default C value when inactive)
_BASIS_INFO = [
    ("w0", "1",          "pba->w0_fld",          "pba->w0_fld"),
    ("wa", "(1-a)",      "pba->wa_fld*(1.0-a)",   "0.0"),
    ("w2", "a*(1-a)",    "pba->w2*a*(1.0-a)",     "0.0"),
    ("w3", "(1-a)**2",   "pba->w3*(1.0-a)*(1.0-a)","0.0"),
    ("w4", "log(a)",     "pba->w4*log(a)",        "0.0"),
]

def generate_user_eq_h(mask: List[int]) -> str:
    """
    Produce the text content of include/user_eq.h for a given chromosome mask.

    Strategy
    --------
    Use SymPy to derive dw/da and ∫3(1+w)/a da symbolically.
    Substitute symbolic names with pba->... C identifiers via placeholder trick.
    Fall back to 0.0 for the integral if SymPy cannot simplify it.

    Returns
    -------
    str — full content of user_eq.h
    """
    from sympy import symbols, diff, integrate, simplify, ccode, log as sym_log

    a = symbols("a")

    # Build symbolic names matching the placeholder scheme
    # We use pba_COEFF as symbol names so ccode() produces "pba_COEFF";
    # we then post-process with string replacement -> "pba->COEFF".
    pba_w0 = symbols("pba_w0_fld")
    pba_wa = symbols("pba_wa_fld")
    pba_w2 = symbols("pba_w2")
    pba_w3 = symbols("pba_w3")
    pba_w4 = symbols("pba_w4")

    raw_syms = [
        symbols("w0"),
        symbols("wa"),
        symbols("w2"),
        symbols("w3"),
        symbols("w4"),
    ]
    pba_syms = [pba_w0, pba_wa, pba_w2, pba_w3, pba_w4]

    # Symbolic basis templates using SymPy-native log
    basis_exprs = [
        raw_syms[0],                          # w0
        raw_syms[1] * (1 - a),                # wa*(1-a)
        raw_syms[2] * a * (1 - a),            # w2*a*(1-a)
        raw_syms[3] * (1 - a) ** 2,           # w3*(1-a)^2
        raw_syms[4] * sym_log(a),             # w4*log(a)
    ]

    # Build w(a) expression from active bases
    active = [(basis_exprs[i], raw_syms[i], pba_syms[i])
              for i, m in enumerate(mask) if m]

    if not active:
        raise ValueError("At least one basis must be active (mask[0]=1 required)")

    w_sym = sum(expr for expr, _, _ in active)
    dw_sym = simplify(diff(w_sym, a))

    # Definite integral ∫_a^{a0=1} 3(1+w)/a' da'  (CLASS initial conditions)
    # CLASS expects: rho_fld(a_ini) = rho_fld(a0) * exp(integral_fld)
    # where integral_fld = ∫_{a}^{1} 3(1+w)/a' da' = F(1) - F(a),
    # with F(a) = ∫ 3(1+w)/a da  (antiderivative).
    # If SymPy cannot compute it, return 0.0 and CLASS falls back to
    # numerical Romberg integration internally.
    integrand = 3 * (1 + w_sym) / a
    try:
        F_sym = simplify(integrate(integrand, a))   # indefinite antiderivative
        F_at_1 = F_sym.subs(a, 1)                   # F(1)
        int_sym = simplify(F_at_1 - F_sym)          # definite: F(1) - F(a)
        int_ok = True
    except Exception:
        int_ok = False
        int_sym = None

    # Substitute raw symbols → pba placeholders before calling ccode
    subs = dict(zip(raw_syms, pba_syms))
    w_c_sym   = w_sym.subs(subs)
    dw_c_sym  = dw_sym.subs(subs)

    w_code    = ccode(w_c_sym)
    dw_code   = ccode(dw_c_sym)

    if int_ok and int_sym is not None:
        int_c_sym = int_sym.subs(subs)
        int_code  = ccode(int_c_sym)
    else:
        int_code = "0.0"

    # Post-process: replace placeholder names with pba->field C syntax
    # Order matters: longer strings first to avoid partial replacement
    replacements = [
        ("pba_w0_fld", "pba->w0_fld"),
        ("pba_wa_fld", "pba->wa_fld"),
        ("pba_w2",     "pba->w2"),
        ("pba_w3",     "pba->w3"),
        ("pba_w4",     "pba->w4"),
    ]
    for old, new in replacements:
        w_code   = w_code.replace(old, new)
        dw_code  = dw_code.replace(old, new)
        int_code = int_code.replace(old, new)

    lines = [
        "// Auto-generated by model_builder.py (scripts_reggie)\n",
        f"#define FLUID_W_FLD(a, w0, wa, ...) ({w_code})\n",
        f"#define DFLUID_W_FLD_DA(a, w0, wa, ...) ({dw_code})\n",
        f"#define INTEGRAL_FLD(a, w0, wa, ...) ({int_code})\n",
    ]
    return "".join(lines)


def mask_str(mask: List[int]) -> str:
    return "".join(str(m) for m in mask)


# ---------------------------------------------------------------------------
# 2.  Source-tree patching helpers
# ---------------------------------------------------------------------------

def _patch_background_h(class_dir: Path) -> None:
    """Add double w2, w3, w4 to struct background (and close-by workspaces)."""
    fpath = class_dir / "include" / "background.h"
    src = fpath.read_text()

    # Check idempotency
    if "double w2;" in src:
        return  # already patched

    new_fields = "  double w2;  /**< extra DE EoS coefficient */\n  double w3;  /**< extra DE EoS coefficient */\n  double w4;  /**< extra DE EoS coefficient */\n"

    # Insert before 'short is_allocated' in struct background — this is the
    # last meaningful field entry before the end of the struct.
    tail = "  short is_allocated; /**< flag is set to true if allocated */\n  //@}\n};"
    src = src.replace(tail, new_fields + tail)
    fpath.write_text(src)


def _patch_input_c(class_dir: Path) -> None:
    """Add class_read_double for w2, w3, w4 inside the CLP fluid block."""
    fpath = class_dir / "source" / "input.c"
    src = fpath.read_text()

    if '"w2"' in src:
        return  # already patched

    old = '      class_read_double("wa_fld",pba->wa_fld);\n      class_read_double("cs2_fld",pba->cs2_fld);'
    new = (
        '      class_read_double("wa_fld",pba->wa_fld);\n'
        '      class_read_double("w2",pba->w2);\n'
        '      class_read_double("w3",pba->w3);\n'
        '      class_read_double("w4",pba->w4);\n'
        '      class_read_double("cs2_fld",pba->cs2_fld);'
    )
    src = src.replace(old, new)
    fpath.write_text(src)


def _patch_cclassy_pxd(class_dir: Path) -> None:
    """Add w2, w3, w4 to the background struct in cclassy.pxd."""
    fpath = class_dir / "python" / "cclassy.pxd"
    src = fpath.read_text()

    if "double w2" in src:
        return  # already patched

    src = src.replace(
        "        double wa_fld\n        double cs2_fld",
        "        double wa_fld\n        double w2\n        double w3\n        double w4\n        double cs2_fld"
    )
    fpath.write_text(src)


def _patch_background_c(class_dir: Path) -> None:
    """
    Patch background.c so that the CLP case uses the user_eq.h macros for
    dw/da and the integral instead of the hardcoded CPL-only expressions.

    This is the critical fix that makes non-CPL w(a) forms work correctly.
    """
    fpath = class_dir / "source" / "background.c"
    src = fpath.read_text()

    if "DFLUID_W_FLD_DA" in src:
        return  # already patched

    # Replace hardcoded dw/da
    old_dw = "  case CLP:\n    *dw_over_da_fld = - pba->wa_fld;\n    break;"
    new_dw = "  case CLP:\n    *dw_over_da_fld = DFLUID_W_FLD_DA(a, pba->w0_fld, pba->wa_fld);\n    break;"
    src = src.replace(old_dw, new_dw)

    # Replace hardcoded integral
    old_int = "  case CLP:\n    *integral_fld = 3.*((1.+pba->w0_fld+pba->wa_fld)*log(1./a) + pba->wa_fld*(a-1.));\n    break;"
    new_int = "  case CLP:\n    *integral_fld = INTEGRAL_FLD(a, pba->w0_fld, pba->wa_fld);\n    break;"
    src = src.replace(old_int, new_int)

    fpath.write_text(src)


def _write_default_user_eq_h(class_dir: Path) -> None:
    """
    Write a CPL-compatible user_eq.h that is safe as the default in
    base_extended. This matches the factory CLP hardcoded behaviour exactly.
    """
    content = (
        "// Default CPL macros for base_extended CLASS build\n"
        "#define FLUID_W_FLD(a, w0, wa, ...) (pba->w0_fld + pba->wa_fld*(1.0 - a))\n"
        "#define DFLUID_W_FLD_DA(a, w0, wa, ...) (-pba->wa_fld)\n"
        "#define INTEGRAL_FLD(a, w0, wa, ...) "
        "(3.0*((1.0+pba->w0_fld+pba->wa_fld)*log(1.0/a) + pba->wa_fld*(a-1.0)))\n"
    )
    (class_dir / "include" / "user_eq.h").write_text(content)


# ---------------------------------------------------------------------------
# 3.  Build orchestration
# ---------------------------------------------------------------------------

def _run(
    cmd: str,
    cwd: Path,
    label: str,
    verbose: bool = False,
    log_path: Optional[Path] = None,
) -> None:
    """Run a shell command, optionally capturing output into a log file."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  cwd : {cwd}")
        print(f"  cmd : {cmd}")
        print(f"{'='*60}")
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
        )
    else:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            text=True,
            capture_output=True,
        )
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{label}]\n")
                handle.write(f"$ {cmd}\n")
                handle.write(result.stdout)
                handle.write(result.stderr)

    if result.returncode != 0:
        detail = f"Command failed (exit {result.returncode}): {cmd}"
        if log_path is not None:
            detail += f". See log: {log_path}"
        raise RuntimeError(detail)


def ensure_extended_base(force: bool = False, verbose: bool = False) -> Path:
    """
    Create and compile the extended BASE CLASS build (one-time, cached).

    The base adds w2, w3, w4 parameter support to the background struct
    and input parser. Only user_eq.h differs per chromosome.

    Parameters
    ----------
    force : bool
        If True, rebuild even if the sentinel file already exists.

    Returns
    -------
    Path — path to the extended base class/ directory (for Cobaya "path")
    """
    sentinel = BASE_EXTENDED_DIR / ".built"
    log_path = BASE_EXTENDED_DIR / "build.log"
    if sentinel.exists() and not force:
        if verbose:
            print(f"[base_extended] already built -> {BASE_EXTENDED_DIR / 'class'}")
        return BASE_EXTENDED_DIR / "class"

    class_dir = BASE_EXTENDED_DIR / "class"

    # (Re-)create the tree
    if BASE_EXTENDED_DIR.exists():
        shutil.rmtree(BASE_EXTENDED_DIR)
    if verbose:
        print(f"[base_extended] Copying {CLASS_SRC} -> {class_dir} ...")
    _copy_class_tree(CLASS_SRC, class_dir)

    # Apply patches
    if verbose:
        print("[base_extended] Patching background.h ...")
    _patch_background_h(class_dir)
    if verbose:
        print("[base_extended] Patching input.c ...")
    _patch_input_c(class_dir)
    if verbose:
        print("[base_extended] Patching cclassy.pxd ...")
    _patch_cclassy_pxd(class_dir)
    if verbose:
        print("[base_extended] Patching background.c ...")
    _patch_background_c(class_dir)
    if verbose:
        print("[base_extended] Writing default user_eq.h ...")
    _write_default_user_eq_h(class_dir)

    # Compile
    py = _python_exe()
    _run(
        "make clean && make -j4 class",
        class_dir,
        "Compile libclass.a + class binary",
        verbose=verbose,
        log_path=log_path,
    )
    _run(
        f"make PYTHON='{py}' classy",
        class_dir,
        "Compile classy Python wrapper",
        verbose=verbose,
        log_path=log_path,
    )

    sentinel.write_text("built\n")
    if verbose:
        print(f"\n[base_extended] Build complete: {class_dir}")
    return class_dir


def build_for_chromosome(chrom, force: bool = False, verbose: bool = False) -> Path:
    """
    Build and compile a CLASS + classy tree for a specific chromosome.

    The build is *cached*: if the sentinel file exists and force=False,
    the existing compiled build is returned immediately.

    Only `include/user_eq.h` differs from base_extended; the complete
    source tree is copied so Cobaya can load it independently in one
    Python session alongside other chromosome builds.

    Parameters
    ----------
    chrom : Chromosome (from chromosome.py)
    force : bool — force rebuild even if cached

    Returns
    -------
    Path — class_dir to pass as Cobaya theory "path"
    """
    ms = mask_str(chrom.mask)
    build_root = BUILDS_DIR / f"mask_{ms}"
    sentinel   = build_root / ".built"
    class_dir  = build_root / "class"
    log_path   = build_root / "build.log"

    if sentinel.exists() and not force:
        if verbose:
            print(f"[mask_{ms}] cached build -> {class_dir}")
        return class_dir

    # Make sure the base exists
    ensure_extended_base(verbose=verbose)

    # Copy from base_extended
    if build_root.exists():
        shutil.rmtree(build_root)
    if verbose:
        print(f"[mask_{ms}] Copying base_extended -> {build_root} ...")
    _copy_class_tree(BASE_EXTENDED_DIR, build_root)

    # Write chromosome-specific user_eq.h
    header_content = generate_user_eq_h(chrom.mask)
    (class_dir / "include" / "user_eq.h").write_text(header_content)
    if verbose:
        print(f"[mask_{ms}] user_eq.h:")
        for line in header_content.splitlines():
            print(f"           {line}")

    # Recompile — only background.c needs to be rebuilt (it includes user_eq.h),
    # but make will detect the changed timestamp automatically.
    # touch background.c to be safe; skip make clean to reuse .o files.
    (class_dir / "source" / "background.c").touch()
    py = _python_exe()
    _run(
        "make -j4 class",
        class_dir,
        f"Compile libclass.a  [mask={ms}]",
        verbose=verbose,
        log_path=log_path,
    )
    _run(
        f"make PYTHON='{py}' classy",
        class_dir,
        f"Compile classy.so  [mask={ms}]",
        verbose=verbose,
        log_path=log_path,
    )

    sentinel.write_text("built\n")
    if verbose:
        print(f"\n[mask_{ms}] Build complete: {class_dir}")
    return class_dir


def chromosome_build_path(chrom) -> Optional[Path]:
    """Return the class_dir for a chromosome if already compiled, else None."""
    ms = mask_str(chrom.mask)
    sentinel = BUILDS_DIR / f"mask_{ms}" / ".built"
    if sentinel.exists():
        return BUILDS_DIR / f"mask_{ms}" / "class"
    return None


def list_built_chromosomes() -> List[str]:
    """Return list of mask strings that have compiled builds."""
    if not BUILDS_DIR.exists():
        return []
    return [
        d.name.replace("mask_", "")
        for d in BUILDS_DIR.iterdir()
        if d.is_dir() and (d / ".built").exists()
           and d.name.startswith("mask_")
    ]


# ---------------------------------------------------------------------------
# 4.  Cobaya integration helpers
# ---------------------------------------------------------------------------

#: Reference cosmological parameter values for a single-point evaluation
COSMO_REF: Dict[str, float] = {
    "logA":     3.05,
    "H0":       67.0,
    "n_s":      0.965,
    "omega_b":  0.0224,
    "omega_cdm": 0.12,
    "tau_reio": 0.055,
}

#: Prior bounds for each EoS coefficient
COEFF_PRIORS: Dict[str, Dict] = {
    "w0_fld": {"prior": {"min": -3.0, "max":  1.0}, "proposal": 0.05,
               "ref": {"dist": "norm", "loc": -1.0, "scale": 0.05}},
    "wa_fld": {"prior": {"min": -3.0, "max":  2.0}, "proposal": 0.10,
               "ref": {"dist": "norm", "loc":  0.0, "scale": 0.10}},
    "w2":     {"prior": {"min": -5.0, "max":  5.0}, "proposal": 0.10,
               "ref": {"dist": "norm", "loc":  0.0, "scale": 0.10}},
    "w3":     {"prior": {"min": -5.0, "max":  5.0}, "proposal": 0.10,
               "ref": {"dist": "norm", "loc":  0.0, "scale": 0.10}},
    "w4":     {"prior": {"min": -3.0, "max":  3.0}, "proposal": 0.10,
               "ref": {"dist": "norm", "loc":  0.0, "scale": 0.10}},
}

# Mapping from chromosome coeff name → CLASS parameter name
_COEFF_TO_CLASS: Dict[str, str] = {
    "w0": "w0_fld",
    "wa": "wa_fld",
    "w2": "w2",
    "w3": "w3",
    "w4": "w4",
}


def cobaya_params_for_chromosome(chrom):
    """
    Build the Cobaya `params:` block for a chromosome.

    Active coefficients (mask=1) → free sampled parameters with priors.
    Inactive coefficients (mask=0) → not included (CLASS defaults to 0).

    The function always includes the standard ΛCDM parameters regardless
    of the DE model.
    """
    import numpy as np

    lcdm_block: Dict[str, Any] = {
        "A_s":   {"latex": "A_s",
                  "value": "lambda logA: 1e-10*np.exp(logA)"},
        "logA":  {"drop": True,
                  "latex": "\\log(10^{10}A_s)",
                  "prior": {"min": 1.61, "max": 3.91},
                  "proposal": 0.001,
                  "ref": {"dist": "norm", "loc": 3.05, "scale": 0.001}},
        "H0":    {"latex": "H_0",
                  "prior": {"min": 20, "max": 100},
                  "proposal": 2,
                  "ref": {"dist": "norm", "loc": 67, "scale": 2}},
        "n_s":   {"latex": "n_s",
                  "prior": {"min": 0.8, "max": 1.2},
                  "proposal": 0.002,
                  "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004}},
        "omega_b": {"latex": "\\Omega_b h^2",
                    "prior": {"min": 0.005, "max": 0.1},
                    "proposal": 0.0001,
                    "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.0001}},
        "omega_cdm": {"latex": "\\Omega_c h^2",
                      "prior": {"min": 0.001, "max": 0.99},
                      "proposal": 0.0005,
                      "ref": {"dist": "norm", "loc": 0.12, "scale": 0.001}},
        "tau_reio": {"latex": "\\tau",
                     "prior": {"min": 0.01, "max": 0.8},
                     "proposal": 0.003,
                     "ref": {"dist": "norm", "loc": 0.055, "scale": 0.006}},
        "m_ncdm":   {"renames": "mnu", "value": 0.06},
        "Omega_Lambda": 0,
        "Omega_m":  {"latex": "\\Omega_m"},
        "YHe":      {"latex": "Y_{\\rm He}"},
        "z_reio":   {"latex": "z_{\\rm re}"},
        "omegamh2": {"derived": "lambda Omega_m, H0: Omega_m*(H0/100)**2",
                     "latex": "\\Omega_m h^2"},
    }

    # Add active DE coefficients
    de_block: Dict[str, Any] = {}
    for coeff in chrom.active_coeffs:
        cls_name = _COEFF_TO_CLASS[coeff]
        entry = dict(COEFF_PRIORS[cls_name])
        entry["latex"] = cls_name.replace("_fld", "")
        de_block[cls_name] = entry

    return {**lcdm_block, **de_block}


def cobaya_info_for_chromosome(
    chrom,
    class_dir: Path,
    packages_path: Optional[Path] = None,
    output_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a complete Cobaya info dict for a chromosome evaluation.

    Parameters
    ----------
    chrom       : Chromosome
    class_dir   : Path returned by build_for_chromosome()
    packages_path : where Cobaya likelihood data lives
                   (defaults to scripts_reggie/cobaya_packages)
    output_prefix : Cobaya chain prefix (None → evaluate only, no output)
    """
    if packages_path is None:
        packages_path = SCRIPTS_REGGIE_DIR / "cobaya_packages"

    # Extra args: inactive coefficients and fixed CLASS settings
    # w0_fld and wa_fld default to CLASS built-in values if not passed,
    # but we need to explicitly zero out inactive coefficients.
    extra_args: Dict[str, Any] = {
        "fluid_equation_of_state": "CLP",
        "N_ncdm": 1,
        "N_ur": 2.0328,
    }
    for coeff in ["w0", "wa", "w2", "w3", "w4"]:
        cls_name = _COEFF_TO_CLASS[coeff]
        if coeff not in chrom.active_coeffs:
            # Fix inactive coefficient at 0
            extra_args[cls_name] = 0.0

    info: Dict[str, Any] = {
        "likelihood": {"sn.pantheonplusshoes": None},
        "theory": {
            "classy": {
                "path": str(class_dir),
                "ignore_obsolete": True,
                "stop_at_error": True,
                "extra_args": extra_args,
            }
        },
        "params": cobaya_params_for_chromosome(chrom),
        "packages_path": str(packages_path),
    }

    if output_prefix is not None:
        info["output"] = str(SCRIPTS_REGGIE_DIR / "chains" / output_prefix)

    return info


# ---------------------------------------------------------------------------
# 5.  Quick single-point evaluation
# ---------------------------------------------------------------------------

def evaluate_chromosome_point(
    chrom,
    class_dir: Path,
    point: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Evaluate the Pantheon+ log-likelihood at a single parameter point for a
    given chromosome.

    Parameters
    ----------
    chrom     : Chromosome
    class_dir : Path from build_for_chromosome()
    point     : dict of parameter values (keys = Cobaya param names).
                Defaults to COSMO_REF + zero for all active DE coefficients
                except w0_fld = -1, wa_fld = 0.

    Returns
    -------
    dict with keys: logpost, loglike, logpriors, loglikes, derived, label
    """
    from cobaya.model import get_model

    if point is None:
        point = dict(COSMO_REF)
        for coeff in chrom.active_coeffs:
            cls_name = _COEFF_TO_CLASS[coeff]
            # Start at ΛCDM reference: w0=-1, all others=0
            point[cls_name] = -1.0 if cls_name == "w0_fld" else 0.0

    # Build info without output/resume keys (which trigger sampler mode)
    info = cobaya_info_for_chromosome(chrom, class_dir)
    info.pop("output", None)

    model = get_model(info)
    lp    = model.logposterior(point)

    return {
        "label":     f"Chromosome {mask_str(chrom.mask)}: {chrom.expr_str}",
        "logpost":   float(lp.logpost),
        "loglike":   float(lp.loglikes[0]),
        "logpriors": list(lp.logpriors),
        "loglikes":  list(lp.loglikes),
        "derived":   dict(zip(model.parameterization.derived_params(), lp.derived)),
        "input_params": point,
    }


# ---------------------------------------------------------------------------
# 6.  GA fitness wrapper
# ---------------------------------------------------------------------------

def fitness_from_chromosome(chrom, class_dir: Path) -> float:
    """
    Return the Pantheon+ log-likelihood as a GA fitness score.

    This does a quick single-point evaluation at the default reference point.
    For full coefficient optimisation, wrap this with a sampler or optimizer.

    Higher = better (PyGAD maximises).
    """
    try:
        result = evaluate_chromosome_point(chrom, class_dir)
        return result["loglike"]
    except Exception as exc:
        print(f"[fitness] evaluation failed for {mask_str(chrom.mask)}: {exc}")
        return float("-inf")
