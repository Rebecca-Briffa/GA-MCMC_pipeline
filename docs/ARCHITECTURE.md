# Architecture

## Process view

```mermaid
flowchart TB
  subgraph driver [GA_driver]
    pygad[PyGAD]
    dna[chromosome.py]
    fit[fitness_metrics.py]
  end
  subgraph workers [Parallel_workers]
    gp[ga_parallel.py]
    cm[client_runner / cobaya_runner]
  end
  subgraph theory [Theory_and_data]
    class[CLASS_classy]
    cobaya[Cobaya_YAML]
    client[CLiENT_TF]
  end
  pygad --> dna
  pygad --> gp
  gp --> cm
  cm --> class
  cm --> cobaya
  cm --> client
  cm --> fit
  fit --> pygad
```

## Core modules (`class/scripts/`)

| Module | Responsibility |
|--------|----------------|
| `chromosome.py` | Shape/coeff pools, mask ↔ expression, repair, seed catalogues |
| `GA_MCMC_new.py` | PyGAD loop, summary I/O, generation-0 seeding, env wiring |
| `fitness_metrics.py` | χ² / AIC / BIC / DIC selection |
| `posterior_diagnostics.py` | DIC from CLiENT samples; unconstrained DE gate |
| `ga_parallel.py` | Multi-GPU / multi-process fitness evaluation |
| `client_runner.py` / `client_worker.py` | CLiENT profiles, GPU placement |
| `cobaya_runner.py` / `cobaya_worker.py` | Cobaya YAML runs |
| `model_builder.py` / `class_modifiers.py` | Inject \(w(a)\) into CLASS user files |
| `params.py` | Shared cosmological / DE parameter helpers |
| `post_ga.py` | Post-search high-accuracy run |

## Config surfaces

- **Environment** — `docker/env.example`; mounted at runtime (not baked into the image).
- **Kubernetes ConfigMap** — `docker/k8s/ga-config.yaml` (`envFrom` on the pod).
- **Cobaya YAMLs** — `class/scripts/cobaya_configs/` (paper default stem: `sn_desi`).
- **CLiENT profiles** — named profiles (`ga_converge`, `ga_final_hpc`, …) selected via `CLIENT_PROFILE` / `GA_POST_PROFILE`.

## Container boundary

The Docker image installs CLASS, Cobaya, CLiENT, and Python deps. **Data packages** (Planck/SN/DESI likelihoods) are prepared with `docker/prepare_data.sh` / `install_cobaya_data.sh` and mounted or layered separately so the Git tree stays small. Working output is always under `GA_OUTPUT_DIR` (PVC or bind mount).
