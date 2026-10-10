# Kubernetes (cosmic-learning) — GA pipeline

Use this after `docker build -t ga-wdm:latest .` on your laptop.

---

## Step 1 — Push image to GHCR

Same registry as `ft-class-builder`:

```bash
# Login once (if needed)
echo $GITHUB_TOKEN | docker login ghcr.io -u rebecca-briffa --password-stdin

# Tag and push
docker tag ga-wdm:latest ghcr.io/rebecca-briffa/ga-wdm:latest
docker push ghcr.io/rebecca-briffa/ga-wdm:latest
```

Verify on GitHub: **Packages** → `ga-wdm`.

---

## Step 2 — Apply config + pod (interactive, like ftclass-pod-max)

Edit runtime choices in **`docker/k8s/ga-config.yaml`** (likelihood, GPU profile, output paths) — not in the Dockerfile.

```bash
kubectl apply -f docker/k8s/ga-config.yaml
kubectl apply -f docker/k8s/ga-pod-interactive.yaml
kubectl get pod -n cosmic-learning ga-wdm-pod
kubectl describe pod -n cosmic-learning ga-wdm-pod   # wait until Running
```

---

## Step 3 — Exec into the pod

```bash
kubectl exec -it -n cosmic-learning ga-wdm-pod -- bash
```

Inside the container:

```bash
# GPU + TensorFlow check
check-gpu

# Create run folder on PVC (persists after pod delete)
mkdir -p /mnt/ga-run

# Full GA (env from ga-config.yaml → /mnt/ga-run)
run-ga
```

Or from your laptop without shelling in:

```bash
kubectl exec -n cosmic-learning ga-wdm-pod -- check-gpu
kubectl exec -n cosmic-learning ga-wdm-pod -- run-ga
```

---

## Step 4 — Watch logs

```bash
kubectl logs -n cosmic-learning ga-wdm-pod -f
```

GA also writes to the PVC:

```bash
kubectl exec -n cosmic-learning ga-wdm-pod -- tail -f /mnt/ga-run/GA_run.log
```

---

## Step 5 — Fetch results

From inside the pod or another mount of `class-pvc`:

```text
/mnt/ga-run/GA_run.log
/mnt/ga-run/GA_summary.txt
/mnt/ga-run/gen_0_model_0/
/mnt/ga-run/gen_1_model_1/
...
```

Copy to laptop (if your cluster allows kubectl cp):

```bash
kubectl cp cosmic-learning/ga-wdm-pod:/mnt/ga-run/GA_run.log ./GA_run.log
```

---

## One-shot Job (no sleep pod)

Runs GA once and exits:

```bash
kubectl apply -f docker/k8s/ga-config.yaml
kubectl delete job -n cosmic-learning ga-wdm-run --ignore-not-found
kubectl apply -f docker/k8s/ga-job-run.yaml
kubectl logs -n cosmic-learning -f job/ga-wdm-run
```

---

## Resource notes (vs your ft-class-builder pod)

| Your old pod | Suggested for GA |
|--------------|------------------|
| 700Gi RAM, 60 CPU, 2 GPU | 64–128Gi RAM, 16–32 CPU, **1 GPU** is enough to start |
| ephemeral-storage 2Gi | **Too small** — use PVC `/mnt/ga-run` for all outputs |
| sleep infinity | Same pattern in `ga-pod-interactive.yaml` |

Scale up if you hit OOM during CLASS compile (many parallel models would need a code change; default GA is **serial** one model at a time).

**2 GPUs:** the GA runs one CLiENT job at a time; a second GPU is unused unless you run two pods/jobs. Use `nvidia.com/gpu: "1"` unless you split work manually.

---

## Environment (edit `ga-config.yaml`)

| Variable | Default | Notes |
|----------|---------|--------|
| `GA_PROJECT_ROOT` | `/mnt/ga-run` | Logs on PVC |
| `GA_OUTPUT_DIR` | `/mnt/ga-run` | `gen_*` dirs on PVC |
| `GA_LIKELIHOOD` | `client` | `client` or `cobaya` |
| `CLIENT_PROFILE` | `ga_fast_gpu` | CLiENT speed/accuracy |
| `GA_CLIENT_GPU` | `auto` | `auto`, `force`, or `cpu` |
| `COBAYA_PACKAGES_PATH` | `/opt/cobaya_packages` | Bundled in image (Option B) |

Local Docker: copy `docker/env.example` and use `docker run --env-file ...`.

---

## Cleanup

```bash
kubectl delete pod -n cosmic-learning ga-wdm-pod
kubectl delete job -n cosmic-learning ga-wdm-run
```

PVC data under `/mnt/ga-run` remains until you delete it on the volume.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ImagePullBackOff` | Check `ghcr-secret`, image name, `docker push` succeeded |
| `no GPU detected` | Node must have GPU + NVIDIA device plugin; pod must request `nvidia.com/gpu` |
| Disk full | Outputs go to PVC; increase PVC size or clean old `gen_*` dirs |
| OOM | Raise `memory` limits in YAML |
