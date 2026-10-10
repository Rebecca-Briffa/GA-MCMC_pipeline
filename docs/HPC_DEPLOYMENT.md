# HPC deployment

## Docker image

```bash
# From repository root
docker/prepare_data.sh          # Cobaya likelihood data (large; not in git)
docker build -f docker/Dockerfile -t ga-wdm:latest .
```

Runtime config is **not** baked in: use `--env-file docker/env.example` or a Kubernetes ConfigMap.

Entrypoint commands (see `docker/entrypoint.sh`): `run-ga`, interactive shell, etc. `run-ga` executes `class/scripts/GA_MCMC_new.py` via `docker/run_ga.sh`.

## GHCR

Tag and push to your registry (example):

```bash
docker tag ga-wdm:latest ghcr.io/<org>/ga-wdm:latest
docker push ghcr.io/<org>/ga-wdm:latest
```

Cluster pull secrets and namespace setup: `docker/DEPLOY_GUIDE.md`, `docker/k8s/README.md`.

## Kubernetes (paper layout)

Namespace used in examples: `cosmic-learning`.

| Manifest | Role |
|----------|------|
| `docker/k8s/ga-config.yaml` | ConfigMap — DIC, pop 20, gens 40, PVC path |
| `docker/k8s/ga-pod-interactive.yaml` | Long-lived pod for `kubectl exec` |
| `docker/k8s/ga-job-run.yaml` | Batch Job |

Typical flow:

```bash
kubectl apply -f docker/k8s/ga-config.yaml
kubectl apply -f docker/k8s/ga-pod-interactive.yaml
kubectl exec -it -n cosmic-learning deploy/... -- bash   # or the named pod
# ensure GA_OUTPUT_DIR exists on the PVC, then:
run-ga
```

**Important:** create the output directory on the PVC before starting (`mkdir -p /mnt/ga-run12` or whatever `GA_OUTPUT_DIR` is). Missing dirs caused early run failures.

Watch progress:

```bash
tail -f "$GA_OUTPUT_DIR/GA_summary.txt"
# or kubectl cp the summary / figures to your laptop
```

## Slurm / bare metal

See `docker/slurm_ga_gpu.sh`, `docker/run_cluster.sh`, and `docker/CLUSTER.md`. Point `GA_CLASS_DIR`, `CLIENT_ROOT`, and `COBAYA_PACKAGES_PATH` at your installs; set `GA_PARALLEL_WORKERS` and `GA_GPU_IDS` to match the allocation.

## Operational gotchas (already fixed in this tree)

| Issue | Fix |
|-------|-----|
| `INITIAL_POP_SIZE` ≠ `GA_POP_SIZE` | Driver sets `INITIAL_POP_SIZE = GA_POP_SIZE` after reading the env |
| Output path missing | Create `GA_OUTPUT_DIR` before `run-ga`; scripts also `mkdir -p` where possible |
| TensorFlow “GPUs: []” at image build | Normal at build time; GPUs appear at runtime with `--gpus` / device plugin |
