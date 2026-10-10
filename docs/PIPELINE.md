# Pipeline

End-to-end path from a binary mask to a ranked generation and (optionally) a post-GA posterior.

## Stages

1. **Encode** — PyGAD holds a bitstring of length `N_BASIS = N_COEFFS × N_SHAPES` (currently 400). Each bit is one (coefficient label × shape function) pair.
2. **Repair** — `chromosome.enforce_max_active_pairs` / unique-shape canonicalisation: keep 2–5 distinct shapes, force the constant \(w_0\) anchor, map each shape to a canonical amplitude label.
3. **Build \(w(a)\)** — Active pairs become an analytic expression string consumed by CLASS / Cobaya parameterisation.
4. **Likelihood** — Default: CLiENT emulator training + Cobaya likelihoods (`GA_LIKELIHOOD=client`, config `sn_desi` = Pantheon+SH0ES + DESI DR2). Optional full Cobaya MCMC backend.
5. **Diagnostics** — `posterior_diagnostics` computes DIC from training log-likelihood samples and flags unconstrained DE coefficients.
6. **Fitness** — `fitness_metrics` selects DIC (default), BIC, AIC, or χ². Failed / unconstrained models receive a sentinel fitness and a `fail_reason` in `GA_summary.txt`.
7. **Evolve** — PyGAD selection, crossover, mutation; parallel workers evaluate one model per GPU/process (`ga_parallel`).
8. **Post-GA** — If `GA_RUN_POST=1`, `post_ga.py` runs a long CLiENT profile (`GA_POST_PROFILE`) and GetDist on the winning `gen_*_model_*` directory.

## Outputs

| Artefact | Role |
|----------|------|
| `GA_summary.txt` | Per-individual rank, metric, mask bits, expression, `fail_reason` |
| `GA_run.log` | Tee’d stdout from the driver |
| `gen_*_model_*/` | Per-model CLASS/Cobaya/CLiENT working dirs |
| Post-GA plots / chains | Under the winning model after `post_ga` |

## Entry points

| Command / script | Role |
|------------------|------|
| `docker/run_ga.sh` → `GA_MCMC_new.py` | Main GA loop |
| `post_ga.py` | Long final chain on best model |
| `plot_best_ga_model.py` | Quick \(w(a)\) / ranking plots |
| `annotate_ga_summary_masks.py` | Back-fill mask columns in older summaries |

See also [ARCHITECTURE.md](ARCHITECTURE.md) and [FITNESS_AND_SELECTION.md](FITNESS_AND_SELECTION.md).

Historical notes from earlier pipeline drafts: [PIPELINE_LEGACY.md](PIPELINE_LEGACY.md).
