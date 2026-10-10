# Fitness and selection

## Metrics (`fitness_metrics.py`)

Lower is better. Selected with `GA_FITNESS_METRIC`:

| Metric | Formula (Gaussian / Cobaya-style) |
|--------|-----------------------------------|
| `chi2` | \(\chi^2\) |
| `aic` | \(\chi^2 + 2k\) |
| `bic` | \(\chi^2 + k\ln N\) |
| `dic` | \(2\bar{D} - \hat{D}\) (**paper default**) |

- \(k\) = number of free Cobaya parameters with priors (cosmology + **active** \(w\) coeffs). Fixed-to-zero unused coeffs are not counted.
- \(N\) = `GA_BIC_N_DATA` (default 1600 ≈ Pantheon+ + DESI BAO order of magnitude) for BIC only.

## DIC (`posterior_diagnostics.py`)

From CLiENT training / posterior log-likelihood samples:

\[
D_i = -2\log L_i,\quad
\bar{D}=\langle D\rangle,\quad
\hat{D}=-2\log L_{\max},\quad
p_D=\bar{D}-\hat{D},\quad
\mathrm{DIC}=\bar{D}+p_D.
\]

DIC rewards fit quality while penalising effective parameter volume—better suited than BIC when parameters are only weakly constrained.

## Unconstrained DE gate

A free dark-energy coefficient is **unconstrained** if its **95%** equal-tailed credible interval covers at least `GA_UNCONSTRAINED_FRAC` (default **0.85**) of the prior width. That catches flat posteriors (e.g. a spurious high-order \(w_6\)) that still achieve a competitive \(\chi^2\).

When `GA_REQUIRE_CONSTRAINED=1` (default):

- such models are marked **failed**;
- `fail_reason` is recorded as `unconstrained:<param>,...` in `GA_summary.txt`;
- fitness is set to a sentinel so PyGAD does not prefer them.

## Selection

PyGAD evolves binary masks subject to repair callbacks (`pygad_on_mutation` and related). Parallel evaluation (`ga_parallel`) returns metric + metadata (`dic`, `unconstrained`, `fail_reason`) into the summary writer in `GA_MCMC_new.py`.
