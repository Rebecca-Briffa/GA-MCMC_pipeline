# Shapes and encoding

## Gene grid

Each individual is a binary mask of length

\[
N_{\mathrm{BASIS}} = N_{\mathrm{coeffs}} \times N_{\mathrm{shapes}} = 20 \times 20 = 400.
\]

Gene \((i,j)\) means: use coefficient label \(w_i\) with shape function \(f_j(a)\). Coefficient *values* are not GA genes—they are fitted by Cobaya/CLiENT for a fixed structure.

## Shape pool (paper set)

Twenty polynomial / mid-\(a\) shapes (constant, \((1-a)\), \(a(1-a)\), higher powers, and selected mixed terms). Log and exponential shapes were dropped after CLASS / pipeline failures on HPC.

Defined in `chromosome.SHAPE_POOL`. Presentation grids live under `results/figures/` (`ga_gene_grid_*.png`, `ga_gene_shapes_*.png`).

## Repair rules

| Rule | Intent |
|------|--------|
| Constant \(w_0\) always on | Anchor every model to a present-day dark-energy amplitude |
| Unique shapes | One amplitude per shape (canonical coeff index) — removes \(w_a\) vs \(w_5\) relabelling duplicates |
| 2–5 active pairs | Pure \(w_0\) excluded (otherwise BIC/DIC collapse to wCDM); complexity capped |
| Max-active truncation | Mutation/crossover excess terms pruned |

After repair, the searchable catalogue (constant on, choose 1–4 additional unique shapes from 19) is on the order of **~5035** models—not \(2^{400}\).

## Expressions

Repaired masks become strings such as

```text
w(a) = w0 + wa*(1-a) + w2*a*(1-a)
```

fed into CLASS user generators (`user_generator_GA.py` / modifiers) and Cobaya parameter blocks.

## Generation 0

`GA_INITIAL_CATALOG` (default `canonical_plus_enumerate`) seeds the first population with literature-like forms plus enumerated short structures, then selects `GA_POP_SIZE` individuals. `INITIAL_POP_SIZE` is forced equal to `GA_POP_SIZE` in `GA_MCMC_new.py`.
