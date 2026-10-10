# Complete Pipeline Guide: Genetic Algorithm Discovery of Dark Energy \(w(a)\)

This document explains the **full wCDM / dark-energy GA pipeline** in `GA_Project-main`: science goal, encoding, genetic algorithm, CLASS integration, Cobaya and CLiENT likelihoods, parallelisation, how to run, outputs, and how to interpret results.

The pipeline is ordinary Python + CLASS. You do **not** need Kubernetes (or Docker) to use it; those are only optional ways to run on a cluster.

It is written so that:

- collaborators can understand the repo from GitHub alone;
- paper drafts can reuse sections with minimal rewriting.

**Main entry point:** `class/scripts/GA_MCMC_new.py`

---

## 1. Scientific goal

Dark energy is often described by an equation of state \(w\), relating pressure and energy density:

\[
w = \frac{P}{\rho}.
\]

Many analyses assume \(w\) is constant (**wCDM**), or evolves slowly with scale factor \(a\) (**CPL**):

\[
w(a) = w_0 + w_a(1-a).
\]

This project asks a broader question:

> Can a genetic algorithm discover a flexible functional form for \(w(a)\) that is supported by supernova data, without fixing the model by hand?

The pipeline therefore:

1. encodes candidate \(w(a)\) forms as binary “chromosomes”;
2. compiles each candidate into the CLASS Boltzmann code;
3. fits free coefficients to **Pantheon+** supernova data (Cobaya and/or CLiENT);
4. ranks models by \(\chi^2\);
5. evolves better structures over generations.

The GA invents **structure** (which terms appear). Cobaya/CLiENT fit **parameter values**.

---

## 2. Big-picture pipeline

```text
┌─────────────────────────────────────────────────────────────────┐
│  Genetic Algorithm (PyGAD)                                       │
│  Evolves binary masks = which coeff×shape terms are ON           │
└───────────────────────────────┬─────────────────────────────────┘
                                │ for each unique mask
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Chromosome → expression                                         │
│  e.g. w(a) = w0 + wa*(1-a) + w4*log(a)                           │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Per-model CLASS build                                           │
│  1. Copy master CLASS tree → gen_N_model_M/class                 │
│  2. SymPy writes user_eq.h (w, dw/da, integral macros)           │
│  3. Patch background.h / .c / input.c / cclassy.pxd              │
│  4. make libclass.a + build model-specific classy.so             │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Likelihood / fitness                                            │
│  Cobaya MCMC  OR  CLiENT emulator (+ optional Cobaya check)      │
│  Data: Pantheon+ (sn.pantheonplus)                               │
│  Fitness = −χ²                                                   │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Selection → crossover → mutation → next generation              │
│  Best model kept (elitism); summaries in GA_summary.txt          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Encoding: how a model is represented

### 3.1 Coefficient × shape free combination

Any candidate is written as a sum of terms:

\[
w(a) = \sum_{(i,j)\in\text{active}} c_i \, f_j(a).
\]

**Coefficients** (free parameters later fitted by MCMC):

| Name | Role |
|------|------|
| `w0` | main constant / anchor term |
| `wa` | often paired with `(1-a)` (CPL-like) |
| `w2` … `w6` | extra free amplitudes |

**Shapes** (functions of scale factor \(a\)):

| Name | Template |
|------|----------|
| `const` | `1` |
| `1ma` | `(1-a)` |
| `1ma2` | `(1-a)^2` |
| `a1ma` | `a*(1-a)` |
| `loga` | `log(a)` |
| `1ma3` | `(1-a)^3` |
| `a2` | `a^2` |
| `1ma4` | `(1-a)^4` |

There are \(7 \times 8 = 56\) possible pairs → **56 binary genes**.

- Gene = 1 → that coefficient×shape term is included.
- Gene = 0 → omitted.

Theoretical search space: \(2^{56}\) masks (not searched exhaustively).

**Cap:** at most `MAX_ACTIVE_PAIRS = 8` terms per model (stability / interpretability).

**Anchor:** `w0 × const` is preferred when trimming masks (wCDM-like baseline).

### 3.2 Important separation

| Layer | What it is | Who decides it |
|-------|------------|----------------|
| Structure | which terms are ON | Genetic algorithm |
| Values | numerical \(w_0, w_a, w_4, \ldots\) | Cobaya / CLiENT MCMC |
| Background cosmology | \(H_0\), \(\omega_b\), \(\omega_{\mathrm{cdm}}\), … | Cobaya priors (`params.py`) |

### 3.3 Canonical starter models (generation-0 seeds)

| Label | Expression | Meaning |
|-------|------------|---------|
| **wCDM** | `w0` | constant dark energy |
| **CPL** | `w0 + wa*(1-a)` | Chevallier–Polarski–Linder |
| **JBP** | `w0 + w2*a*(1-a)` | Jassal–Bagla–Padmanabhan-like |
| **poly2** | `w0 + wa*(1-a) + w2*a*(1-a)` | two dynamical terms |
| **log** | `w0 + w4*log(a)` | logarithmic correction |

Default initial catalogue: `canonical_plus_enumerate` — these five plus additional unique random masks until population size is filled.

Implemented in: `class/scripts/chromosome.py`.

---

## 4. Genetic algorithm settings

Configured in `GA_MCMC_new.py` (defaults):

| Setting | Default | Meaning |
|---------|---------|---------|
| Population size | 10 | models per generation |
| Generations | 20 | evolution length after gen-0 scan |
| Parents mating | 8 | how many parents enter crossover |
| Crossover | uniform @ 95% | each gene randomly from parent A or B |
| Mutation | 50% of genes | random bit flips on offspring |
| Parent selection | `sss` | steady-state selection |
| Elitism | 1 | keep best individual unchanged |
| Fitness | \(-\chi^2\) | higher fitness = better fit |

### 4.1 One generation, step by step

1. **Decode** each binary mask → expression string.
2. **Evaluate** new masks (cache hits skip re-runs).
3. **Rank** by fitness \(-\chi^2\).
4. **Select** parents.
5. **Crossover** to form children.
6. **Mutate** children (then enforce max active terms / dedupe).
7. **Elitism** copies the best parent into the next population.
8. Log generation block to `GA_summary.txt`.

### 4.2 Caching and duplicates

- `_fitness_cache`: mask → (fitness, likelihood backend). Same structure is never re-evaluated.
- Duplicate masks within a generation are skipped (fitness \(-\infty\)).

---

## 5. From expression to CLASS cosmology

For each unevaluated model `gen_N_model_M`:

### 5.1 Directory preparation (`run_model_results.py`)

- Create `GA_OUTPUT_DIR/gen_N_model_M/class/` by copying the master CLASS tree (`GA_CLASS_DIR`).
- Old folder with the same name is removed first (clean rebuild).

### 5.2 Analytic macros (`user_generator_GA.py`)

Given expression e.g. `w0+wa*(1-a)+w4*log(a)`, SymPy builds:

- `FLUID_W_FLD` — \(w(a)\)
- `DFLUID_W_FLD_DA` — \(\mathrm{d}w/\mathrm{d}a\)
- `INTEGRAL_FLD` — background integral needed by CLASS for fluid energy density

Written to `include/user_eq.h`.

### 5.3 CLASS source patches (`class_modifiers.py`)

Any new coefficient beyond `{a, w0, wa}` (e.g. `w4`) is added to:

- `background.h` struct field
- `background.c` initialisation
- `input.c` parameter parser (`class_read_double`)
- `python/cclassy.pxd` Cython binding

Also patches CLP branches so derivative and integral use the custom macros (not hardcoded CPL formulas).

### 5.4 Compile

1. `make class libclass.a` — C library for this model.
2. `python setup.py build` in `python/` — model-specific `classy.so`.

Cobaya must load **this** `classy`, not a global install (`classy_path.py` prepends the model build path).

---

## 6. Likelihood backends (how \(\chi^2\) is obtained)

Controlled by:

```bash
export GA_LIKELIHOOD=cobaya   # or client
```

### 6.1 Cobaya MCMC (`GA_LIKELIHOOD=cobaya`)

- Theory: CLASS with `fluid_equation_of_state: CLP` (custom macros).
- Likelihood: Pantheon+ (`sn.pantheonplus`).
- Sampler: Cobaya MCMC (default ~200 samples in GA mode — exploration, not final publication chains).
- Extra fluid parameters get priors in `params.py` / `extend_params()`.
- Runs in a **subprocess** (`cobaya_worker.py`) so each model loads a clean `classy`.

Fitness uses best `chi2__SN` from the chain (or recovered from partial chain files if MCMC stops early).

### 6.2 CLiENT emulator (`GA_LIKELIHOOD=client`)

Used for faster GA exploration on GPU clusters:

1. Draw ~400 Latin-hypercube true Cobaya+CLASS evaluations.
2. Train a neural network surrogate of the likelihood (TensorFlow).
3. Run fast emcee on the surrogate.
4. Fitness from best **true** χ² among training evaluations (optionally re-validated with one Cobaya call).

Profiles (`CLIENT_PROFILE`):

| Profile | Behaviour |
|---------|-----------|
| `ga_fast` | 1 training iteration; fastest exploration |
| `ga_fast_gpu` | same + larger batches for GPU |
| `ga_converge` | Gelman–Rubin R−1 stopping |
| `ga_accurate` | 3 fixed iterations |

Requires `CLIENT_ROOT` pointing to the CLiENT repository.

**Important:** CLiENT does **not** remove CLASS compile cost. It mainly replaces long Cobaya MCMC loops.

### 6.3 What runs on CPU vs GPU

| Step | Hardware |
|------|----------|
| CLASS `make` / Cython build | CPU |
| True Cobaya likelihood evals | CPU |
| CLiENT NN training | GPU (when available) |
| Emulator emcee | mostly CPU / light GPU use |

---

## 7. Parallelisation

Implemented in `ga_parallel.py`.

### 7.1 What is parallel

**Multiple models in the same generation** can be evaluated at once.

Each parallel job runs the full per-model pipeline (CLASS copy → compile → likelihood).

Environment:

```bash
export GA_PARALLEL_WORKERS=2
export GA_GPU_IDS=0,1
```

- Default in code: `GA_PARALLEL_WORKERS=1` (serial).
- Set `GA_PARALLEL_WORKERS=2` (and `GA_GPU_IDS=0,1` if you have two GPUs) when you want parallel model evaluation.

Mechanism: `ProcessPoolExecutor` with **spawn** (CUDA/TensorFlow safe). Each worker may set `CUDA_VISIBLE_DEVICES` to one GPU (round-robin).

### 7.2 What is not parallel

| Part | Parallel? |
|------|-----------|
| Generations | No — sequential |
| GA operators (selection/crossover/mutation) | No — after fitnesses known |
| Cobaya chains inside one model | No — `COBAYA_NOMPI=1` |

### 7.3 Mental picture

```text
Generation N
  Model 0 ─┐
  Model 1 ─┼─ up to GA_PARALLEL_WORKERS simultaneous evaluations
  Model 2 ─┘
       ↓
  all χ² known → selection / crossover / mutation
       ↓
Generation N+1
```

---

## 8. Key Python modules

| File | Role |
|------|------|
| `GA_MCMC_new.py` | Main GA loop, fitness, logging |
| `chromosome.py` | Encoding, canonical models, mutation hooks |
| `run_model_results.py` | Copy CLASS, compile, build classy |
| `user_generator_GA.py` | SymPy → `user_eq.h` macros |
| `class_modifiers.py` | Inject new parameters into CLASS sources |
| `params.py` | Cobaya ΛCDM + fluid parameter priors |
| `cobaya_runner.py` / `cobaya_worker.py` | Cobaya MCMC χ² |
| `client_runner.py` / `client_worker.py` | CLiENT emulator χ² |
| `ga_parallel.py` | Multi-model parallel evaluation |
| `classy_path.py` | Prefer per-model `classy.so` |

Legacy / older scripts (`GA_MCMC.py`, etc.) exist but **`GA_MCMC_new.py` is the current entry point**.

---

## 9. Environment variables (runtime configuration)

| Variable | Typical value | Meaning |
|----------|---------------|---------|
| `GA_CLASS_DIR` | `.../class` | Master CLASS template |
| `GA_PROJECT_ROOT` | project root (default) | Where logs are written |
| `GA_OUTPUT_DIR` | same as project root | Where `gen_*` folders are written |
| `GA_LIKELIHOOD` | `cobaya` (default) or `client` | Likelihood backend |
| `CLIENT_ROOT` | path to CLiENT | Required for client mode |
| `CLIENT_PROFILE` | `ga_fast` or `ga_fast_gpu` | CLiENT speed/accuracy profile |
| `COBAYA_PACKAGES_PATH` | Cobaya data root | Pantheon+ packages |
| `COBAYA_NOMPI` | `1` | Serial Cobaya (no MPI) |
| `GA_PARALLEL_WORKERS` | `1` or `2` | Parallel model evaluations |
| `GA_GPU_IDS` | `0,1` | GPU assignment for workers |
| `GA_CLIENT_GPU` | `auto` | Auto-detect / use GPU for TF |

Export these in your shell before running `python GA_MCMC_new.py`.

---

## 10. How to run

The normal way is to run the Python entry point on a machine with CLASS dependencies and Cobaya data. Docker/Kubernetes are optional packaging for clusters.

### 10.1 Local run with Cobaya (simplest)

```bash
cd class/scripts
export COBAYA_NOMPI=1
export COBAYA_PACKAGES_PATH=/path/to/cobaya_packages
python GA_MCMC_new.py
```

### 10.2 Local run with CLiENT (faster exploration)

```bash
cd class/scripts
export COBAYA_NOMPI=1
export COBAYA_PACKAGES_PATH=/path/to/cobaya_packages
export GA_LIKELIHOOD=client
export CLIENT_ROOT=/path/to/client_public
export CLIENT_PROFILE=ga_fast          # or ga_fast_gpu if you have a GPU
python GA_MCMC_new.py
```

Optional parallel models:

```bash
export GA_PARALLEL_WORKERS=2
export GA_GPU_IDS=0,1
```

### 10.3 Optional: Docker

If you prefer a containerised environment:

```bash
docker build -f docker/Dockerfile -t ga-wdm:latest .
docker run --rm --gpus all -v $PWD/runs:/work ga-wdm:latest run-ga
```

This is optional convenience, not required for the science pipeline.

### 10.4 Optional: cluster / Kubernetes

Some sites wrap the same image in Kubernetes Jobs or interactive pods. That only changes *where* `GA_MCMC_new.py` runs, not the algorithm. See `docker/` only if you need that deployment path.

---

## 11. Outputs and how to read them

By default, outputs appear under the project root (`GA_PROJECT_ROOT` / `GA_OUTPUT_DIR`).

| Output | Contents |
|--------|----------|
| `GA_run.log` | Full terminal log (settings, compiles, MCMC/CLiENT progress) |
| `GA_summary.txt` | Per-generation table: model ID, χ², fitness, backend, expression |
| `gen_N_model_M/` | Full CLASS copy, chains / CLiENT runs for that individual |

**Lower χ² is better.** Fitness printed by the GA is \(-\chi^2\).

Typical published/exploration result in this codebase: models near

\[
w(a) = w_0 + w_a(1-a) + w_4\log a
\]

often compete closely with pure wCDM, with χ² around the Pantheon+-only minimum found by the run (exact numbers depend on data cuts and sampler settings).

---

## 12. Cosmology interpretation (for papers)

- Scale factor \(a=1\) today; redshift \(z = 1/a - 1\).
- Pantheon+ constrains luminosity distances → expansion history → \(w(a)\).
- This pipeline uses **background distances** (non-linear clustering off for SN-only fits).
- Discovered expressions should be compared to standard benchmarks: **wCDM**, **CPL**, and optionally **JBP**.
- Extra terms (e.g. \(\log a\)) are only physically interesting if they improve fit enough to justify extra parameters (information criteria / nested comparisons can be added in analysis, not automatic in the GA fitness).

---

## 13. Design choices and limitations

**Strengths**

- Automatic structure search over a large basis of \(w(a)\) terms.
- Each candidate is a real CLASS cosmology, not a toy distance fitter alone.
- Swap between accurate Cobaya MCMC and fast CLiENT exploration.
- Can run locally; optional Docker packaging for reproducible environments.

**Limitations**

- SN-only by default (extendable to BAO/CMB in Cobaya info).
- GA MCMC settings are for **search**, not final posterior quality.
- CLASS compile per model dominates wall-time.
- Large theoretical gene space; exploration is stochastic, not exhaustive.
- Emulator profiles trade accuracy for speed.

---

## 14. Roadmap / next goals (research narrative)

1. Interpret best \(f(a)/w(a)\) models as candidate dark-energy histories and compare rigorously with wCDM and CPL.
2. Extend the GA to search over **scalar-field potentials** \(V(\phi)\).
3. Map each potential through the scalar-field equations to \(w(a)\) or energy-density evolution.
4. Reuse the same CLASS + data-fitting + GA stack so potential forms (exponential, power-law, combinations) can be discovered rather than chosen by hand.
5. Build toward GA-driven model selection for dark energy and scalar-field–inspired modified gravity.

---

## 15. One-paragraph summary (abstract-ready)

This pipeline searches for a flexible dark-energy equation of state \(w(a)\) by combining simple building blocks. Each candidate is encoded as a binary chromosome of coefficient×shape genes; a gene turned on includes that term in the model. A genetic algorithm starts from standard seeds (wCDM, CPL, JBP, …) and evolves new structures with selection, crossover, mutation, and elitism. For each candidate, the pipeline copies CLASS, injects the analytic \(w(a)\) (and its derivative and integral) into the Boltzmann code, compiles a model-specific library, and scores the model against Pantheon+ supernovae using Cobaya MCMC and/or the CLiENT neural likelihood emulator. Fitness is \(-\chi^2\). Optional parallel workers evaluate several models per generation on multiple GPUs. Results are logged in `GA_summary.txt` and per-model `gen_*` directories, providing an automated route from structure search to cosmological model comparison.

---

## 16. Quick file map for GitHub readers

```text
GA_Project-main/
├── class/                      # Master CLASS + pipeline scripts
│   └── scripts/
│       ├── GA_MCMC_new.py      # ← start here
│       ├── chromosome.py
│       ├── run_model_results.py
│       ├── cobaya_*.py
│       ├── client_*.py
│       └── ga_parallel.py
├── docker/                     # Optional container packaging
├── docs/                       # Longer LaTeX guides (optional)
└── PIPELINE.md                 # This document
```
