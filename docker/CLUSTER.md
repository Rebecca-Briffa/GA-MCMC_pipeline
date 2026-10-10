# Running the GA pipeline on an HPC cluster (Docker / GPU)

This folder packages the **mask-based GA + CLASS + Cobaya + CLiENT** pipeline for
supercomputers. Unlike the minimal CLASS-only image you showed, this image:

- **Keeps** the full CLASS source tree (the GA copies it for every `gen_*_model_*`)
- Includes **CLiENT** + **TensorFlow GPU** for emulator training
- Installs **Pantheon+** via Cobaya (or bundles your full `cobaya_packages`)
- Writes all run output to a **mounted volume** (`/work`)

---

## 1. What to put on GitHub

Do **not** push `gen_*` model directories (gigabytes of CLASS builds).

```bash
cd GA_Project-main
bash docker/prepare_github_repo.sh ../ga-wdm-github
cd ../ga-wdm-github
git init
git add .
git commit -m "GA w(a) discovery pipeline with Docker"
git remote add origin https://github.com/YOURUSER/ga-wdm.git
git push -u origin main
```

**Include in the repo:**

| Path | Purpose |
|------|---------|
| `class/` | CLASS template + `class/scripts/` pipeline |
| `docker/` | Dockerfile, Slurm script, helpers |
| `docs/` | Guides (optional) |

**Exclude (handled by `.dockerignore` / export script):**

- `gen_*` run directories
- `GA_run.log`, `GA_summary.txt`
- `docker/data/cobaya_packages/` (too large for GitHub)

**For datasets on GitHub:** use one of:

1. **Build-time download** (default) — Pantheon+ only, ~160 MB, no upload needed
2. **Cluster filesystem** — rsync `cobaya_packages` to `$SCRATCH`, bind-mount at runtime
3. **Git LFS** — if your site requires a frozen data bundle in the repo

---

## 2. Bundle datasets (optional)

### Pantheon+ only (recommended for SN-only GA)

Nothing to do — the Dockerfile runs `cobaya-install sn.pantheonplusshoes` (+ DESI) automatically.

### Full local Cobaya packages (~2 GB)

On your laptop:

```bash
cd GA_Project-main
bash docker/prepare_data.sh /home/rebecca/Project_hiclass/cobaya_packages copy
# or tarball:
bash docker/prepare_data.sh /home/rebecca/Project_hiclass/cobaya_packages tar
docker build -f docker/Dockerfile -t ga-wdm:latest .
```

---

## 3. Build the image

On a machine with Docker and network access:

```bash
cd GA_Project-main   # or your GitHub clone
docker build -f docker/Dockerfile -t ga-wdm:latest .
```

Build time: ~20–40 min (CLASS compile + TensorFlow + Cobaya data).

### Convert to Singularity (most HPC sites)

On the cluster login node (after `docker save` on build machine or build on cluster if allowed):

```bash
# On build machine:
docker save ga-wdm:latest | gzip > ga-wdm.tar.gz
scp ga-wdm.tar.gz cluster:~/

# On cluster:
gunzip -c ga-wdm.tar.gz | singularity build ga-wdm.sif docker-archive:/dev/stdin
# or if docker is available on cluster:
singularity build ga-wdm.sif docker-daemon://ga-wdm:latest
```

---

## 4. Run with GPU

### Interactive test

```bash
export GA_WORK=$SCRATCH/ga-test
mkdir -p $GA_WORK

docker run --rm -it --gpus all \
  -v $GA_WORK:/work \
  ga-wdm:latest check-gpu

docker run --rm -it --gpus all \
  -v $GA_WORK:/work \
  ga-wdm:latest run-ga
```

Output appears in `$GA_WORK`:

- `GA_run.log`, `GA_summary.txt`
- `gen_0_model_0/`, `gen_1_model_1/`, …

### Singularity

```bash
export GA_WORK=$SCRATCH/ga-test
mkdir -p $GA_WORK

singularity exec --nv \
  -B $GA_WORK:/work \
  --env GA_LIKELIHOOD=client \
  --env CLIENT_PROFILE=ga_fast_gpu \
  --env GA_CLIENT_GPU=auto \
  ga-wdm.sif run-ga
```

### Slurm batch job

Edit `docker/slurm_ga_gpu.sh` (partition, account, modules), then:

```bash
sbatch docker/slurm_ga_gpu.sh
```

Or use the helper:

```bash
export GA_WORK=$SCRATCH/ga-run-001
export GA_IMAGE=ga-wdm:latest
bash docker/run_cluster.sh
```

---

## 5. Environment variables

Runtime choices are **not** baked into the image. Defaults live in `docker/entrypoint.sh`; override via:

- **Kubernetes:** `docker/k8s/ga-config.yaml` (ConfigMap)
- **Docker:** `docker/env.example` → `docker run --env-file ...`
- **Slurm / shell:** `-e VAR=value` or export before `run-ga`

| Variable | Default (entrypoint) | Meaning |
|----------|----------------------|---------|
| `GA_PROJECT_ROOT` | `/work` | Logs (`GA_run.log`) |
| `GA_OUTPUT_DIR` | `/work` | `gen_*` model directories |
| `GA_CLASS_DIR` | `/opt/ga-project/class` | Read-only CLASS template |
| `CLIENT_ROOT` | `/opt/client_public` | CLiENT install |
| `COBAYA_PACKAGES_PATH` | `/opt/cobaya_packages` | Likelihood data |
| `GA_LIKELIHOOD` | `client` | `client` or `cobaya` |
| `CLIENT_PROFILE` | `ga_fast_gpu` | CLiENT profile |
| `GA_CLIENT_GPU` | `auto` | GPU for TensorFlow |
| `COBAYA_NOMPI` | `1` | Serial Cobaya (no MPI) |

Mount external Cobaya data:

```bash
docker run --gpus all \
  -v $SCRATCH/cobaya_packages:/opt/cobaya_packages:ro \
  -v $GA_WORK:/work \
  ga-wdm:latest run-ga
```

---

## 6. What runs on GPU vs CPU

| Step | Hardware |
|------|----------|
| CLASS `make` per model | CPU |
| 400 LHS Pantheon+ evals | CPU |
| CLiENT NN training | **GPU** (TensorFlow) |
| Cobaya validation (1 eval) | CPU |

The GPU mainly speeds CLiENT training; CLASS compile dominates total time.

---

## 7. Differences from your example Dockerfile

Your example **deleted** `/opt/CLASS` after installing a global `classy` package.
This pipeline **cannot** do that because each GA individual:

1. Copies the full CLASS tree
2. Patches `user_eq.h`, `background.c`, …
3. Recompiles `libclass.a` and model-specific `classy.so`

The template stays at `GA_CLASS_DIR=/opt/ga-project/class`; outputs go to `/work`.

---

## 8. Troubleshooting

| Problem | Fix |
|---------|-----|
| `no GPU detected` in container | Run with `--gpus all` (Docker) or `--nv` (Singularity); check `nvidia-smi` on node |
| `CLIENT_ROOT not set` | Should be set in image; check you use this Dockerfile |
| Pantheon+ not found | Re-run with `-e COBAYA_INSTALL_ON_START=1` or mount `cobaya_packages` |
| Out of GPU memory | `-e CLIENT_PROFILE=ga_fast` (smaller batches) or `-e GA_CLIENT_GPU=cpu` |
| Permission on `/work` | `mkdir -p $SCRATCH/ga-run && chmod u+rwx` |

---

## 9. Quick reference

```bash
# Build
docker build -f docker/Dockerfile -t ga-wdm:latest .

# GPU check
docker run --rm --gpus all ga-wdm:latest check-gpu

# Full GA run
mkdir -p $SCRATCH/ga-run
docker run --rm --gpus all -v $SCRATCH/ga-run:/work ga-wdm:latest run-ga
```
