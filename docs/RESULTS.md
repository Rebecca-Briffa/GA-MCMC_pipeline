# Results

Curated artefacts under `results/` (not full `gen_*` trees).

## Summaries (`results/summaries/`)

| File | Content |
|------|---------|
| `GA_summary_ga-run10.txt` | Earlier BIC-era run summary |
| `GA_summary_ga-run11*.txt` | Later run + mask-annotated variants |
| `ga_run10_best_model_table.txt` | Best-model table excerpt |

A live run writes `GA_summary.txt` into `GA_OUTPUT_DIR`. Columns typically include generation, rank, fitness metric, status, `fail_reason`, mask bitstring, and \(w(a)\) expression.

`fail_reason` values of interest:

- `unconstrained:<params>` — posterior gate failed
- other pipeline errors — CLASS/Cobaya/CLiENT failures (see matching `gen_*` logs)

## Figures (`results/figures/`)

Presentation and paper-support plots, including:

- Gene grids and shape illustrations (`ga_gene_grid_*.png`, `ga_gene_shapes_*.png`)
- Pipeline flowchart (`ga_pipeline_flowchart.png`)
- Run-10/11 rankings, \(\Delta\chi^2\), \(w(z)\) / \(w(a)\) profiles, literature comparison
- Canonical seed tables

These are static exports for slides and the README narrative; regenerate from a finished run with `plot_best_ga_model.py` and related scripts.

## ΛCDM baseline

`lcdm_baseline/` holds SN+DESI reference YAMLs and runners for comparing GA winners against a fixed ΛCDM likelihood stack—not evolutionary search output.
