# CLiENT integration for the GA pipeline

[CLiENT](https://github.com/AarhusCosmology/client_public) (Cosmological Likelihood Emulator using Neural networks with TensorFlow) trains a neural network on **true** Cobaya+CLASS likelihood calls, then runs fast MCMC on the surrogate. The GA uses the best point from CLiENT’s **true** training cloud (Latin hypercube), validated with one extra Cobaya evaluation for `chi2__SN`.

## Important constraints

1. **Per GA model you still compile CLASS** — each mask is a different `w(a)` and parameter set. CLiENT does not remove compile cost; it replaces the long Cobaya MCMC loop (200+ samples) with ~400 true evals + cheap emcee on the NN (profile `ga_fast`).

2. **First-time setup is large** — CLiENT’s `setup.sh` creates a conda env with TensorFlow, Cobaya, and optional Planck data (~8 GB). Use the same Cobaya/Pantheon+ packages you already use for the GA.

3. **Accuracy vs speed** — `ga_fast` is tuned for exploration; use `ga_accurate` or full Cobaya for final models.

## CLASS / classy conflicts in `clienv`

CLiENT setup may `pip install` a **global** `classy` from your GA CLASS tree. Cobaya must load the **per-model** `classy.so` built under `gen_*_model_*/class/python/build/...`.

The GA scripts now prepend that path automatically. If problems persist:

```bash
conda activate clienv
pip uninstall classy -y
```

Do **not** reinstall classy globally in `clienv`; each GA model builds its own wrapper.

## MPI error (`cannot load MPI library`)

If Cobaya/CLiENT crash with `mpi4py` / `libmpi.so` errors, you do **not** need MPI for serial GA runs. Either:

```bash
export COBAYA_NOMPI=1
```

or install a working MPI stack in `clienv`:

```bash
conda activate clienv
conda install -y -c conda-forge openmpi mpi4py
```

The GA scripts set `COBAYA_NOMPI=1` by default when you run `GA_MCMC_new.py`.

## Install CLiENT

```bash
cd /home/rebecca/CLASS   # or your preferred parent directory
git clone https://github.com/AarhusCosmology/client_public.git
cd client_public
./setup.sh
```

Follow the prompts (Cobaya backend, paths to CLASS, etc.).

## Point the GA at CLiENT

In the shell where you run the GA (same env as Cobaya + CLASS):

```bash
export CLIENT_ROOT=/home/rebecca/CLASS/client_public
export GA_LIKELIHOOD=client          # use CLiENT instead of Cobaya MCMC
export CLIENT_PROFILE=ga_fast      # or ga_accurate
export CLIENT_VALIDATE_MAP=1       # one true Cobaya eval at best training point (recommended)
```

Then run the GA as usual:

```bash
cd /home/rebecca/CLASS/GA_Project-main/class/scripts
python GA_MCMC_new.py
```

If `CLIENT_ROOT` is unset or CLiENT fails, the pipeline **falls back** to `run_cobaya_best_chi2`.

Each model is tagged in `GA_summary.txt` with **`likelihood_backend`**:

| Value | Meaning |
|-------|---------|
| `client` | CLiENT ran successfully |
| `cobaya_fallback` | CLiENT failed; Cobaya MCMC used instead |
| `cobaya` | Direct Cobaya (`GA_LIKELIHOOD=cobaya`) |
| `cobaya_recovered` | Error during eval; chi2 recovered from saved chain |
| `failed` | No usable chi2 |
| `skipped_duplicate` | Duplicate mask in same generation |

## Profiles

| Profile       | CLiENT stopping | Typical cost per model |
|---------------|-----------------|-------------------------|
| `ga_fast`     | **Fixed 1 iteration** (`-i 1`) — no R−1 loop | Fastest |
| `ga_fast_gpu` | Same as `ga_fast`, **batch_size 512** for GPU   | Use with NVIDIA + TF CUDA |
| `ga_converge` | **Gelman–Rubin R−1 < 0.01** (up to 20 iterations) | Standard cosmology |
| `ga_accurate` | Fixed 3 iterations (`-i 3`) | Middle ground |
| `ga_final`    | R−1 loop, **T=1**, more LHS samples, longer emcee | **Post-GA winner only** |

### Post-GA final pass (recommended for GetDist)

`post_ga.py` is a **separate** script (not inside the GA loop). It rebuilds the
winner (optional), runs CLiENT with `ga_final` (T=1, denser training, longer
emcee), then writes GetDist + param tables under `post_ga_plots/`.

`GA_MCMC_new.py` calls it automatically at the end unless you set
`export GA_RUN_POST=0`.

Standalone (e.g. your already-copied best model):

```bash
export CLIENT_ROOT=/home/rebecca/CLASS/client_public
conda activate clienv
cd /home/rebecca/CLASS/GA_Github_0708/class/scripts
python post_ga.py /path/to/gen_16_model_9 \
  --expr 'w0+w0*(1-a)^2+w2*a*(1-a)+w3*a*(1-a)+w3*(1-a)^4+w4+w4*(1-a)^3+w5*(1-a)^3' \
  --ga-chi2 1410.4787
```

Or rebuild from scratch: `python post_ga.py --rebuild --expr '...'`.

Set the profile when running the GA:

```bash
export CLIENT_PROFILE=ga_converge
```

Or force convergence mode on any profile:

```bash
export CLIENT_USE_CONVERGENCE=1
```

### What `-i 1` means

CLiENT’s flag **`-i N`** = “run exactly **N** training iterations, then stop.”

When `-i` is set, CLiENT **disables** Gelman–Rubin convergence checking.  
When `-i` is **omitted**, CLiENT loops: train → sample → resample → check **R−1** until R−1 < threshold or `max_iterations`.

Our `ga_fast` profile passes `-i 1` for speed. **`ga_converge` does not pass `-i`**, so you get standard R−1-based stopping.

Edit `GA_CLIENT_PROFILES` in `client_runner.py` to tune further.

## GPU acceleration (TensorFlow / CLiENT)

CLiENT trains a Keras neural network with **TensorFlow**. If `clienv` was built with
`tensorflow[and-cuda]` (CLiENT’s default), TF automatically uses an NVIDIA GPU when
available. Your RTX 4060 Laptop GPU is already detected in past runs.

**What uses the GPU vs CPU**

| Step | Hardware | Notes |
|------|----------|--------|
| CLASS compile + 400 LHS likelihood calls | **CPU** | Dominates wall time per model |
| CLiENT NN training (250 epochs) | **GPU** | Main benefit of enabling GPU |
| CLiENT emcee on surrogate | Mostly **CPU** | Cheap compared to CLASS |
| Cobaya validation (1 call) | **CPU** | One CLASS eval at best point |

### Quick check

```bash
conda activate clienv
export CLIENT_ROOT=/home/rebecca/CLASS/client_public
cd /home/rebecca/CLASS/GA_Project-main/class/scripts
python check_client_gpu.py
```

### Recommended environment for GPU runs

```bash
conda activate clienv
export CLIENT_ROOT=/home/rebecca/CLASS/client_public
export GA_LIKELIHOOD=client
export CLIENT_PROFILE=ga_fast_gpu      # larger batches for GPU
export GA_CLIENT_GPU=auto                # default: use GPU if TF sees one
export COBAYA_NOMPI=1
export COBAYA_PACKAGES_PATH=/home/rebecca/Project_hiclass/cobaya_packages
python GA_MCMC_new.py
```

With `GA_CLIENT_GPU=auto` and `CLIENT_PROFILE=ga_fast`, the pipeline **automatically
upgrades** to `ga_fast_gpu` when a GPU is detected (disable with
`export GA_CLIENT_AUTO_GPU_PROFILE=0`).

### GPU environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GA_CLIENT_GPU` | `auto` | `auto`, `1`/`force`, or `0`/`cpu` |
| `CUDA_VISIBLE_DEVICES` | (unset) | Set to `0` to pick first GPU; pipeline sets `-1` if `GA_CLIENT_GPU=cpu` |
| `CLIENT_TF_GPU_ALLOW_GROWTH` | `1` | Maps to `TF_FORCE_GPU_ALLOW_GROWTH=true` (saves VRAM on laptop GPUs) |
| `CLIENT_TF_LOG_LEVEL` | `2` | Less TensorFlow stderr spam (`0` = verbose) |
| `GA_CLIENT_AUTO_GPU_PROFILE` | `1` | Auto `ga_fast` → `ga_fast_gpu` when GPU present |
| `CLIENT_GPU_PROFILE` | `ga_fast_gpu` | Profile name used for auto-upgrade |

Implementation: `client_gpu.py` (env + probe), applied in `client_worker.py` before
each `client.py` subprocess.

### Troubleshooting GPU

- **No GPU in probe but `nvidia-smi` works** — reinstall TF in clienv:
  `pip install 'tensorflow[and-cuda]==2.19.0'` (match `client_public/environment.yaml`).
- **Out-of-memory on laptop GPU** — keep `CLIENT_TF_GPU_ALLOW_GROWTH=1`; use
  `ga_fast` (batch 128) instead of `ga_fast_gpu`; or `export GA_CLIENT_GPU=cpu`.
- **WSL2** — ensure Windows NVIDIA driver + WSL CUDA support; run `nvidia-smi` inside WSL.

## Outputs per model

Under each model directory:

- `scripts/client_config/<model>.yaml` — Cobaya input for that CLASS build  
- `scripts/client_config/<model>_client.yaml` — CLiENT master config  
- `scripts/client_runs/<timestamp>_<model>/` — scalers, NN, training_data HDF5, chains  

## Cobaya-only mode (default)

```bash
export GA_LIKELIHOOD=cobaya
# or unset GA_LIKELIHOOD
```

## References

- Paper: [arXiv:2512.17509](https://arxiv.org/abs/2512.17509)  
- Repo: https://github.com/AarhusCosmology/client_public  
