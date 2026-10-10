"""
chromosome.py
=============
Building blocks for a GA that discovers dark-energy equations of state.

Architecture
------------
The GA evolves *structure* via a binary mask over coefficient × shape pairs::

    w(a) = sum over active (coeff_i, shape_j) of  coeff_i * shape_j(a)

Paper encoding (unique-shape repair)
------------------------------------
* 20 polynomial / mid-``a`` shapes in ``SHAPE_POOL`` (no log/exp).
* 20 coefficient labels in ``COEFF_POOL``; raw grid is 20×20 = 400 genes.
* Repair enforces: constant ``w0`` anchor, one amplitude per shape,
  ``MIN_ACTIVE_PAIRS``–``MAX_ACTIVE_PAIRS`` (2–5) unique shapes.
* Searchable catalogue after repair ≈ few thousand models (not 2^400).

Coefficient *values* are not GA genes; Cobaya/CLiENT fit them once the
structure is fixed.

PyGAD integration
-----------------
    gene_space  = [[0, 1]] * N_BASIS   (N_BASIS = N_COEFFS * N_SHAPES)
    fitness_func → Chromosome from mask → expr_str → likelihood

Public API
----------
    COEFF_POOL, SHAPE_POOL, N_COEFFS, N_SHAPES, N_BASIS, COEFF_NAMES
    MAX_ACTIVE_PAIRS, enforce_max_active_pairs
    Chromosome, mask_to_chromosome, init_population, ...
    mask_to_bitstring, bitstring_to_mask, mask_from_expr
"""

from __future__ import annotations

import random
from itertools import product
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# 1. Coefficient and shape pools (free combination)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShapeFunction:
    """A scale-factor-dependent shape f(a) with no coefficient attached."""
    name:     str
    f:        Callable[[np.ndarray], np.ndarray]
    template: str   # e.g. "(1-a)", "log(a)" — omit coefficient prefix


# One amplitude label per shape (canonical unique-shape repair).
COEFF_POOL: List[str] = [
    "w0", "wa", "w2", "w3", "w4", "w5", "w6", "w7", "w8", "w9",
    "w10", "w11", "w12", "w13", "w14", "w15", "w16", "w17", "w18", "w19",
]

SHAPE_POOL: List[ShapeFunction] = [
    # 20 poly / mid-a shapes (unique-shape + canonical coeffs).
    # Avoided: log(a), exp(1-a)-1 (CLASS / pipeline failures on HPC).
    ShapeFunction("const",     lambda a: np.ones_like(a),
                  "1"),
    ShapeFunction("1ma",       lambda a: 1.0 - a,
                  "(1-a)"),
    ShapeFunction("a1ma",      lambda a: a * (1.0 - a),
                  "a*(1-a)"),
    ShapeFunction("1ma2",      lambda a: (1.0 - a) ** 2,
                  "(1-a)^2"),
    ShapeFunction("a1ma2",     lambda a: a * (1.0 - a) ** 2,
                  "a*(1-a)^2"),
    ShapeFunction("1ma3",      lambda a: (1.0 - a) ** 3,
                  "(1-a)^3"),
    ShapeFunction("a21ma",     lambda a: (a ** 2) * (1.0 - a),
                  "a^2*(1-a)"),
    ShapeFunction("a1ma3",     lambda a: a * (1.0 - a) ** 3,
                  "a*(1-a)^3"),
    ShapeFunction("1ma4",      lambda a: (1.0 - a) ** 4,
                  "(1-a)^4"),
    ShapeFunction("a21ma2",    lambda a: (a ** 2) * (1.0 - a) ** 2,
                  "a^2*(1-a)^2"),
    ShapeFunction("a31ma",     lambda a: (a ** 3) * (1.0 - a),
                  "a^3*(1-a)"),
    ShapeFunction("a21ma3",    lambda a: (a ** 2) * (1.0 - a) ** 3,
                  "a^2*(1-a)^3"),
    ShapeFunction("a31ma2",    lambda a: (a ** 3) * (1.0 - a) ** 2,
                  "a^3*(1-a)^2"),
    ShapeFunction("a1ma4",     lambda a: a * (1.0 - a) ** 4,
                  "a*(1-a)^4"),
    ShapeFunction("a41ma",     lambda a: (a ** 4) * (1.0 - a),
                  "a^4*(1-a)"),
    ShapeFunction("1ma5",      lambda a: (1.0 - a) ** 5,
                  "(1-a)^5"),
    ShapeFunction("a31ma3",    lambda a: (a ** 3) * (1.0 - a) ** 3,
                  "a^3*(1-a)^3"),
    ShapeFunction("a41ma2",    lambda a: (a ** 4) * (1.0 - a) ** 2,
                  "a^4*(1-a)^2"),
    ShapeFunction("a21ma4",    lambda a: (a ** 2) * (1.0 - a) ** 4,
                  "a^2*(1-a)^4"),
    ShapeFunction("a1m2a1ma",  lambda a: a * (1.0 - 2.0 * a) * (1.0 - a),
                  "a*(1-2*a)*(1-a)"),
]

N_COEFFS = len(COEFF_POOL)
N_SHAPES = len(SHAPE_POOL)
N_GENES = N_COEFFS * N_SHAPES
N_BASIS = N_GENES          # PyGAD alias (one gene per coeff×shape pair)
COEFF_NAMES = list(COEFF_POOL)

# w0 paired with the constant shape — required anchor in every model
ANCHOR_PAIR = (0, 0)

# Pure w0×1 excluded (MIN=2); BIC otherwise locks onto wCDM.
MIN_ACTIVE_PAIRS = 2
# Cap complexity; unique-shape rule still applies
MAX_ACTIVE_PAIRS = 5

# Above this gene count, full binary enumeration is impractical
MAX_EXACT_ENUMERATION = 20


def pair_index(coeff_idx: int, shape_idx: int) -> int:
    """Flat gene index for coefficient ``coeff_idx`` × shape ``shape_idx``."""
    return coeff_idx * N_SHAPES + shape_idx


def unpack_pair(flat_idx: int) -> Tuple[int, int]:
    """Return ``(coeff_idx, shape_idx)`` for a flat gene index."""
    return divmod(flat_idx, N_SHAPES)


ANCHOR_FLAT = pair_index(*ANCHOR_PAIR)


def active_indices_from_mask(mask: Sequence[int]) -> List[int]:
    """Return flat indices where the mask is active."""
    return [i for i, v in enumerate(mask) if v]


def _canonical_coeff_for_shape(shape_idx: int) -> int:
    """Fixed amplitude label per shape (removes wa vs w5 relabeling duplicates)."""
    if shape_idx == ANCHOR_PAIR[1]:
        return ANCHOR_PAIR[0]
    return min(shape_idx, N_COEFFS - 1)


def enforce_max_active_pairs(mask: Sequence[int],
                             max_pairs: int = MAX_ACTIVE_PAIRS,
                             rng: Optional[random.Random] = None,
                             *,
                             min_pairs: int = MIN_ACTIVE_PAIRS,
                             force_anchor: bool = True) -> List[int]:
    """
    Repair a mask to a canonical unique-shape model:

    - at most one active pair per shape
    - const shape uses w0 only (anchor)
    - shape ``s`` uses coefficient ``_canonical_coeff_for_shape(s)``
    - between ``min_pairs`` and ``max_pairs`` active shapes
    """
    out_active_shapes: set[int] = set()
    for i, v in enumerate(mask):
        if int(v):
            _c, s = unpack_pair(i)
            if 0 <= s < N_SHAPES:
                out_active_shapes.add(s)

    if force_anchor:
        out_active_shapes.add(ANCHOR_PAIR[1])

    shapes = sorted(out_active_shapes)

    # Cap number of shapes (always keep const).
    if len(shapes) > max_pairs:
        keep = [ANCHOR_PAIR[1]] if force_anchor else []
        remaining = [s for s in shapes if s not in keep]
        slots = max_pairs - len(keep)
        if slots > 0:
            if rng is None:
                keep.extend(remaining[:slots])
            else:
                keep.extend(rng.sample(remaining, min(slots, len(remaining))))
        shapes = sorted(keep)

    # Floor: need dynamical shape(s) beyond const.
    while len(shapes) < min_pairs:
        candidates = [s for s in range(N_SHAPES) if s not in shapes]
        if not candidates:
            break
        if rng is None:
            prefer = list(range(1, N_SHAPES))
            pick = next((s for s in prefer if s in candidates), candidates[0])
        else:
            pick = rng.choice(candidates)
        shapes.append(pick)
        shapes = sorted(shapes)

    repaired = [0] * N_GENES
    for s in shapes:
        c = _canonical_coeff_for_shape(s)
        repaired[pair_index(c, s)] = 1
    return repaired

def pairs_to_mask(pairs: Sequence[Tuple[int, int]]) -> List[int]:
    """Build a flat mask from a list of ``(coeff_idx, shape_idx)`` pairs."""
    mask = [0] * N_GENES
    for c_idx, s_idx in pairs:
        if not (0 <= c_idx < N_COEFFS and 0 <= s_idx < N_SHAPES):
            raise ValueError(f"invalid pair ({c_idx}, {s_idx})")
        mask[pair_index(c_idx, s_idx)] = 1
    return mask


def _term_string(coeff: str, shape: ShapeFunction) -> str:
    if shape.template == "1":
        return coeff
    return f"{coeff}*{shape.template}"


# ---------------------------------------------------------------------------
# 2. Chromosome
# ---------------------------------------------------------------------------

@dataclass
class Chromosome:
    """
    Binary mask over coeff×shape pairs encoding one w(a) functional form.

    Parameters
    ----------
    mask : list of 0/1, length N_GENES
        mask[pair_index(c, s)] == 1  →  include COEFF_POOL[c] * SHAPE_POOL[s]

    Derived attributes
    ------------------
    active_indices  : flat gene indices that are on
    active_pairs    : (coeff_idx, shape_idx) for each active gene
    active_coeffs   : unique coefficient names used (MCMC parameters)
    expr_str        : e.g. "w(a)=w0+wa*(1-a)"
    n_params        : len(active_coeffs)
    """

    mask: List[int]
    active_indices: List[int]       = field(init=False)
    active_pairs:   List[Tuple[int, int]] = field(init=False)
    active_coeffs:  List[str]       = field(init=False)
    expr_str:       str             = field(init=False)
    n_params:       int             = field(init=False)

    def __post_init__(self):
        if len(self.mask) != N_GENES:
            raise ValueError(f"mask must have length {N_GENES}, got {len(self.mask)}")

        self.active_indices = [i for i, v in enumerate(self.mask) if v]
        self.active_pairs = [unpack_pair(i) for i in self.active_indices]
        seen: set[str] = set()
        self.active_coeffs = []
        for c_idx, _ in self.active_pairs:
            name = COEFF_POOL[c_idx]
            if name not in seen:
                seen.add(name)
                self.active_coeffs.append(name)
        self.n_params = len(self.active_coeffs)
        self.expr_str = self._build_expr_str()

    def _build_expr_str(self) -> str:
        if not self.active_pairs:
            return "w(a)=0"
        terms = [
            _term_string(COEFF_POOL[c], SHAPE_POOL[s])
            for c, s in self.active_pairs
        ]
        return "w(a)=" + "+".join(terms)

    def evaluate(self, a: np.ndarray, coeffs: dict) -> np.ndarray:
        """Evaluate w(a) for active coeff×shape pairs."""
        a = np.asarray(a, dtype=float)
        w = np.zeros_like(a)
        for c_idx, s_idx in self.active_pairs:
            coeff = COEFF_POOL[c_idx]
            w += coeffs[coeff] * SHAPE_POOL[s_idx].f(a)
        return w

    def pair_summary(self) -> str:
        """Compact description of active coeff×shape pairs."""
        if not self.active_pairs:
            return "(empty)"
        parts = [
            f"{COEFF_POOL[c]}×{SHAPE_POOL[s].name}"
            for c, s in self.active_pairs
        ]
        return ", ".join(parts)

    def __repr__(self) -> str:
        return f"Chromosome(pairs={self.pair_summary()}, expr='{self.expr_str}')"


# ---------------------------------------------------------------------------
# 3. Population helpers
# ---------------------------------------------------------------------------

def mask_to_chromosome(mask: Sequence[int]) -> Chromosome:
    """Construct a Chromosome from a raw flat mask (≤ MAX_ACTIVE_PAIRS terms)."""
    return Chromosome(mask=enforce_max_active_pairs(mask))


def _all_model_masks(include_zero: bool = True) -> List[List[int]]:
    """Enumerate every binary mask (only when N_GENES is small enough)."""
    if N_GENES > MAX_EXACT_ENUMERATION:
        raise ValueError(
            f"exact enumeration needs N_GENES<={MAX_EXACT_ENUMERATION}, "
            f"got {N_GENES} ({N_COEFFS} coeffs × {N_SHAPES} shapes)"
        )
    masks = [list(mask) for mask in product((0, 1), repeat=N_GENES)]
    if include_zero:
        return masks
    return [mask for mask in masks if any(mask)]


def _random_pair_mask(rng: random.Random,
                      *,
                      anchor: bool = True,
                      pair_prob: float = 0.15) -> List[int]:
    """Random unique-shape mask with ``MIN``…``MAX`` active shapes (canonical coeffs)."""
    del pair_prob
    n_terms = rng.randint(MIN_ACTIVE_PAIRS, min(MAX_ACTIVE_PAIRS, N_SHAPES))
    shapes = {0} if anchor else set()
    while len(shapes) < n_terms:
        shapes.add(rng.randrange(N_SHAPES))
    mask = [0] * N_GENES
    for s in shapes:
        mask[pair_index(_canonical_coeff_for_shape(s), s)] = 1
    return enforce_max_active_pairs(mask, rng=rng, force_anchor=anchor)


def _random_unique_masks(n: int,
                         rng: random.Random,
                         *,
                         seeds: Optional[Sequence[Sequence[int]]] = None,
                         include_zero: bool = False) -> List[List[int]]:
    """Return ``n`` unique masks, starting with ``seeds`` then random draws."""
    seen: set[tuple[int, ...]] = set()
    ordered: List[List[int]] = []
    for raw in seeds or ():
        repaired = enforce_max_active_pairs(raw, rng=rng)
        key = tuple(repaired)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(repaired)
    attempts = 0
    max_attempts = max(1000, n * 200)
    while len(ordered) < n and attempts < max_attempts:
        attempts += 1
        candidate = _random_pair_mask(rng)
        if not include_zero and not any(candidate):
            continue
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
    if len(ordered) < n:
        raise RuntimeError(
            f"could only sample {len(ordered)} unique masks (requested {n})"
        )
    return ordered[:n]


def _canonical_seed_population() -> List[Chromosome]:
    """Two fixed seeds for the random initialiser (CPL, JBP)."""
    seeds = canonical_seed_masks()
    return [mask_to_chromosome(seeds[0]), mask_to_chromosome(seeds[1])]


# Shape idx: 0=const,1=1ma,2=a1ma,3=1ma2,4=a1ma2,5=1ma3,6=a21ma
# Canonical coeffs: shape s → coeff s (w0…w6). No pure wCDM.
CANONICAL_FIVE_PAIRS: List[List[Tuple[int, int]]] = [
    [(0, 0), (1, 1)],                       # CPL
    [(0, 0), (2, 2)],                       # JBP
    [(0, 0), (1, 1), (2, 2)],               # poly2
    [(0, 0), (3, 3)],                       # (1-a)^2
    [(0, 0), (1, 1), (4, 4)],               # CPL + a(1-a)^2
]

CANONICAL_FIVE_MASKS = [pairs_to_mask(p) for p in CANONICAL_FIVE_PAIRS]


def pad_mask_to_basis(mask: Sequence[int]) -> List[int]:
    """Extend a shorter legacy mask with zeros (for imports of old 5/10-bit masks)."""
    bits = [int(v) for v in mask]
    if len(bits) > N_GENES:
        raise ValueError(f"mask length {len(bits)} exceeds N_GENES={N_GENES}")
    return bits + [0] * (N_GENES - len(bits))


def canonical_seed_masks() -> List[List[int]]:
    """Five named models as flat coeff×shape masks."""
    return [list(m) for m in CANONICAL_FIVE_MASKS]


def canonical_five_population() -> List[Chromosome]:
    """Five named DE models for the first GA generation."""
    return [mask_to_chromosome(m) for m in canonical_seed_masks()]


def _dedupe_masks(preferred: Sequence[Sequence[int]],
                  rest: Sequence[Sequence[int]]) -> List[List[int]]:
    """Merge mask lists, keeping first occurrence order."""
    seen: set[tuple[int, ...]] = set()
    ordered: List[List[int]] = []
    for mask in (*preferred, *rest):
        key = tuple(int(v) for v in mask)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(list(key))
    return ordered


def masks_for_initial_catalog(catalog: str,
                              size: Optional[int] = None,
                              *,
                              include_zero_model: bool = False,
                              rng: Optional[random.Random] = None) -> List[List[int]]:
    """
    Return up to ``size`` binary masks for a named initial catalogue.

    With coeff×shape encoding (N_GENES = N_COEFFS × N_SHAPES), full
    enumeration is only available when N_GENES <= MAX_EXACT_ENUMERATION.
    Larger spaces use the five canonical models plus random unique masks.
    """
    if rng is None:
        rng = random.Random()

    name = catalog.strip().lower()
    seeds = canonical_seed_masks()
    can_enumerate = N_GENES <= MAX_EXACT_ENUMERATION
    full_size = (2 ** N_GENES) if include_zero_model else ((2 ** N_GENES) - 1)

    if name in ("full", "enumerate"):
        if can_enumerate:
            all_masks = _all_model_masks(include_zero=include_zero_model)
            ordered = _dedupe_masks(seeds, all_masks)
            if name == "full":
                if size is not None and size != full_size:
                    raise ValueError(f"catalog 'full' requires size={full_size} or size=None")
                return ordered
            if size is None:
                size = full_size
            return ordered[:size]
        # Large search space: canonical seeds + random fill
        if size is None:
            size = len(seeds) if name == "canonical_five" else min(10, 50)
        return _random_unique_masks(size, rng, seeds=seeds,
                                      include_zero=include_zero_model)

    if name == "canonical_five":
        if size is not None and size != len(seeds):
            raise ValueError(f"catalog 'canonical_five' requires size={len(seeds)}")
        return seeds

    if name == "canonical_plus_enumerate":
        if can_enumerate:
            all_masks = _all_model_masks(include_zero=include_zero_model)
            ordered = _dedupe_masks(seeds, all_masks)
            if size is None:
                size = len(ordered)
            if size > len(ordered):
                raise ValueError(
                    f"catalog 'canonical_plus_enumerate' has at most {len(ordered)} "
                    f"unique masks (N_GENES={N_GENES}), got size={size}"
                )
            return ordered[:size]
        if size is None:
            size = min(10, 50)
        return _random_unique_masks(size, rng, seeds=seeds,
                                    include_zero=include_zero_model)

    raise ValueError(
        f"unknown initial_catalog {catalog!r}; expected one of "
        "canonical_five, canonical_plus_enumerate, enumerate, full"
    )


def _parse_model_name(model_name: str) -> List[int]:
    """Resolve short labels like 'wcdm', 'cpl', or explicit pair specs."""
    name = model_name.strip().lower()
    aliases = {
        "zero": pairs_to_mask([]),
        "empty": pairs_to_mask([]),
        "none": pairs_to_mask([]),
        "w0": pairs_to_mask([(0, 0)]),
        "w0_only": pairs_to_mask([(0, 0)]),
        "wcdm": pairs_to_mask([(0, 0)]),
        "cpl": pairs_to_mask([(0, 0), (1, 1)]),
        "w0wa": pairs_to_mask([(0, 0), (1, 1)]),
        "jbp": pairs_to_mask([(0, 0), (2, 2)]),
        "poly2": pairs_to_mask([(0, 0), (1, 1), (2, 2)]),
        "log": pairs_to_mask([(0, 0), (4, 3)]),
    }
    if name in aliases:
        return aliases[name]

    if name.endswith("_only"):
        coeff = name[:-5]
        if coeff in COEFF_POOL:
            c_idx = COEFF_POOL.index(coeff)
            return pairs_to_mask([(c_idx, 0)])

    cleaned = name.replace("+", " ").replace(",", " ").replace("_", " ")
    parts = [part for part in cleaned.split() if part]
    if parts and all(part in COEFF_POOL for part in parts):
        return pairs_to_mask([(COEFF_POOL.index(p), 0) for p in parts])

    raise ValueError(f"unrecognised flat_model specification: {model_name!r}")


def _coerce_model_mask(model: str | Sequence[int] | Chromosome) -> List[int]:
    """Convert a flat-model specification into a binary mask."""
    if isinstance(model, Chromosome):
        return list(model.mask)
    if isinstance(model, str):
        return _parse_model_name(model)

    mask = [int(v) for v in model]
    if len(mask) != N_GENES:
        raise ValueError(f"flat_model mask must have length {N_GENES}, got {len(mask)}")
    if any(bit not in (0, 1) for bit in mask):
        raise ValueError("flat_model mask entries must be binary (0 or 1)")
    return mask


def _export_expr(chrom: Chromosome) -> str:
    """Render an expression string suitable for the MODEL/EXPR text export."""
    if not chrom.active_pairs:
        return "0"
    expr = chrom.expr_str
    if expr.startswith("w(a)="):
        return expr.split("=", 1)[1].strip()
    return expr


def _canonical_model_names() -> dict[tuple[int, ...], str]:
    """Return canonical export names for selected masks."""
    return {
        tuple(CANONICAL_FIVE_MASKS[0]): "CPL",
        tuple(CANONICAL_FIVE_MASKS[1]): "JBP",
        tuple(CANONICAL_FIVE_MASKS[2]): "poly2",
        tuple(CANONICAL_FIVE_MASKS[3]): "w0_1ma2",
        tuple(CANONICAL_FIVE_MASKS[4]): "CPL_a1ma2",
    }


def _model_label(chrom: Chromosome) -> str:
    """Build a stable base label for a chromosome in the export file."""
    mask_tuple = tuple(chrom.mask)
    canonical_names = _canonical_model_names()
    if mask_tuple in canonical_names:
        return canonical_names[mask_tuple]
    active = sum(chrom.mask)
    return f"mask_{active}p_{''.join(str(bit) for bit in chrom.mask)}"


def _write_population_catalog(population: Sequence[Chromosome], output_path: str) -> None:
    """Write the initial population to a MODEL/EXPR catalogue file."""
    coeff_args = ", ".join(["a", *COEFF_NAMES])
    lines = [
        "/* user_eq.txt — coeff×shape free combination */",
        f"/* coeffs: {', '.join(COEFF_POOL)} */",
        f"/* shapes: {', '.join(s.name for s in SHAPE_POOL)} */",
        f"/* #define FLUID_W_FLD({coeff_args}) (...) */",
        "",
    ]
    seen_labels: dict[str, int] = {}
    for chrom in population:
        base_label = _model_label(chrom)
        seen_labels[base_label] = seen_labels.get(base_label, 0) + 1
        label = base_label if seen_labels[base_label] == 1 else f"{base_label}_{seen_labels[base_label]}"
        lines.append(f"MODEL={label}")
        lines.append(f"EXPR={_export_expr(chrom)}")
        lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def random_chromosome(rng: Optional[random.Random] = None) -> Chromosome:
    """Generate a random Chromosome with the w0×1 anchor on."""
    if rng is None:
        rng = random.Random()
    return Chromosome(mask=_random_pair_mask(rng, anchor=True))


def init_population(size: Optional[int] = None,
                    seed: Optional[int] = None,
                    random_pop: bool = False,
                    initial_catalog: str = "enumerate",
                    flat_model: Optional[str | Sequence[int] | Chromosome] = None,
                    include_zero_model: bool = False,
                    output_path: Optional[str] = None) -> List[Chromosome]:
    """
    Build an initial population of chromosomes.

    Catalogues
    ----------
    canonical_five
        CPL, JBP, poly2, w0+(1-a)^2, CPL+a(1-a)^2 (size must be 5).
        Pure wCDM omitted; unique shapes + canonical coeffs; BIC ranks complexity.
    canonical_plus_enumerate
        Those five first, then more unique masks until ``size``.
    enumerate / full
        Exact binary order when N_GENES is small; otherwise canonical five
        plus random unique masks.
    """
    catalog = initial_catalog.strip().lower()
    can_enumerate = N_GENES <= MAX_EXACT_ENUMERATION
    full_size = (2 ** N_GENES) if include_zero_model else ((2 ** N_GENES) - 1)
    default_sizes = {
        "canonical_five": len(CANONICAL_FIVE_MASKS),
        "canonical_plus_enumerate": min(10, full_size if can_enumerate else 50),
        "enumerate": full_size if can_enumerate else min(10, 50),
        "full": full_size if can_enumerate else min(10, 50),
    }
    if catalog not in default_sizes:
        raise ValueError(
            f"initial_catalog must be one of {sorted(default_sizes)}, got {initial_catalog!r}"
        )
    if size is None:
        size = default_sizes[catalog]
    if size <= 0:
        raise ValueError("size must be a positive integer")

    rng = random.Random(seed)

    if flat_model is not None:
        template_mask = _coerce_model_mask(flat_model)
        population = [mask_to_chromosome(template_mask) for _ in range(size)]
    elif random_pop:
        population = _canonical_seed_population()
        while len(population) < size:
            population.append(random_chromosome(rng))
        population = population[:size]
    else:
        masks = masks_for_initial_catalog(
            catalog, size, include_zero_model=include_zero_model, rng=rng
        )
        population = [mask_to_chromosome(mask) for mask in masks]

    if output_path is not None:
        _write_population_catalog(population, output_path)

    return population


# ---------------------------------------------------------------------------
# 4. GA operators
# ---------------------------------------------------------------------------

def mask_key(mask) -> tuple:
    """Hashable key for a binary mask."""
    return tuple(int(v) for v in mask)


def mask_to_bitstring(mask: Sequence[int]) -> str:
    """Compact gene mask for GA_summary (e.g. ``1001010...`` length N_GENES)."""
    return "".join("1" if int(v) else "0" for v in mask)


def bitstring_to_mask(bits: str) -> List[int]:
    """Parse a bitstring from GA_summary back to a length-N_GENES mask."""
    bits = bits.strip()
    if not bits or any(c not in "01" for c in bits):
        raise ValueError(f"invalid gene mask bitstring: {bits!r}")
    if len(bits) != N_GENES:
        raise ValueError(
            f"gene mask length {len(bits)} != N_GENES={N_GENES}"
        )
    return [int(c) for c in bits]


def mask_from_expr(expr: str) -> List[int]:
    """
    Reconstruct the canonical unique-shape mask from a ``w(a)=...`` expression.

    Terms must match current SHAPE_POOL templates with canonical coeffs.
    """
    raw = (expr or "").strip()
    if raw.startswith("w(a)="):
        raw = raw.split("=", 1)[1].strip()
    if not raw or raw == "0":
        return enforce_max_active_pairs([0] * N_GENES)

    # Split on '+' outside parentheses (shapes have no nested '+').
    parts: List[str] = []
    buf = ""
    depth = 0
    for ch in raw:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "+" and depth == 0:
            if buf.strip():
                parts.append(buf.strip())
            buf = ""
            continue
        buf += ch
    if buf.strip():
        parts.append(buf.strip())

    template_to_shape = {s.template: i for i, s in enumerate(SHAPE_POOL)}
    pairs: List[Tuple[int, int]] = []
    for term in parts:
        if "*" in term:
            coeff, shape_tmpl = term.split("*", 1)
            coeff = coeff.strip()
            shape_tmpl = shape_tmpl.strip()
        else:
            coeff = term.strip()
            shape_tmpl = "1"
        if shape_tmpl not in template_to_shape:
            raise ValueError(f"unknown shape template in term {term!r}")
        s_idx = template_to_shape[shape_tmpl]
        c_idx = _canonical_coeff_for_shape(s_idx)
        # Accept either the canonical name or any coeff; repair will canonicalize.
        if coeff not in COEFF_POOL and coeff not in ("w0", "wa"):
            # still allow; pairing by shape only
            pass
        pairs.append((c_idx, s_idx))
    return enforce_max_active_pairs(pairs_to_mask(pairs))


def flip_mask_genes(mask: Sequence[int],
                    gene_indices: Sequence[int],
                    rng: Optional[random.Random] = None) -> List[int]:
    """XOR-flip selected genes, then enforce the term cap."""
    out = [int(v) for v in mask]
    for i in gene_indices:
        out[i] ^= 1
    return enforce_max_active_pairs(out, rng=rng)


def random_mask_mutation(mask: Sequence[int],
                         rng: Optional[random.Random] = None,
                         n_flips: int = 1) -> List[int]:
    """Flip ``n_flips`` genes (any position in the flat mask)."""
    if rng is None:
        rng = random.Random()
    n_flips = min(max(1, n_flips), N_GENES)
    picks = rng.sample(list(range(N_GENES)), n_flips)
    return flip_mask_genes(mask, picks, rng=rng)


def crossover(parent_a: Chromosome,
              parent_b: Chromosome,
              rng: Optional[random.Random] = None) -> tuple[Chromosome, Chromosome]:
    """Single-point crossover on the flat mask."""
    if rng is None:
        rng = random.Random()

    point = rng.randint(1, N_GENES - 1)
    child_a_mask = parent_a.mask[:point] + parent_b.mask[point:]
    child_b_mask = parent_b.mask[:point] + parent_a.mask[point:]
    return mask_to_chromosome(child_a_mask), mask_to_chromosome(child_b_mask)


def mutate(chrom: Chromosome,
           rng: Optional[random.Random] = None,
           flip_prob: float = 0.25) -> Chromosome:
    """Bit-flip mutation on any gene in the mask."""
    if rng is None:
        rng = random.Random()

    new_mask = list(chrom.mask)
    for i in range(N_GENES):
        if rng.random() < flip_prob:
            new_mask[i] ^= 1
    return mask_to_chromosome(new_mask)


def repair_duplicate_offspring(offspring,
                               reserved_masks: set,
                               rng: Optional[random.Random] = None,
                               max_attempts: int = 32):
    """Ensure each offspring row has a unique mask not in ``reserved_masks``."""
    import numpy as np

    if rng is None:
        rng = random.Random()

    out = np.array(offspring, dtype=int).copy()
    for row_idx in range(out.shape[0]):
        out[row_idx] = enforce_max_active_pairs(out[row_idx], rng=rng)
        key = mask_key(out[row_idx])
        attempts = 0
        while key in reserved_masks and attempts < max_attempts:
            out[row_idx] = random_mask_mutation(out[row_idx], rng=rng, n_flips=1)
            key = mask_key(out[row_idx])
            attempts += 1
        reserved_masks.add(key)
    return out


def pygad_on_mutation(ga, offspring):
    """PyGAD ``on_mutation`` hook: cap terms, then dedupe offspring masks vs. elitism."""
    import numpy as np

    rng = random.Random()
    if getattr(ga, "random_seed", None) is not None:
        rng.seed(ga.random_seed + int(getattr(ga, "generations_completed", 0)))

    out = np.array(offspring, dtype=int).copy()
    for row_idx in range(out.shape[0]):
        out[row_idx] = enforce_max_active_pairs(out[row_idx], rng=rng)

    reserved: set = set()

    if ga.keep_elitism and ga.last_generation_fitness is not None:
        elitism, _ = ga.steady_state_selection(
            ga.last_generation_fitness, ga.keep_elitism
        )
        for row in elitism:
            reserved.add(mask_key(row))

    return repair_duplicate_offspring(out, reserved, rng=rng)
