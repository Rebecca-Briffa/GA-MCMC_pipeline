# GA w(a) pipeline — build, push, and test (step-by-step)

Repeatable workflow from laptop → GitHub → Docker image → GHCR → Kubernetes cluster.

**Repo root for all commands:** `/home/rebecca/CLASS/GA_Project-main`

---

## Before you start

| Requirement | Notes |
|-------------|--------|
| Docker | WSL2 or Linux with Docker installed |
| Cobaya data (optional) | Full bundle at `/home/rebecca/Project_hiclass/cobaya_packages` |
| GitHub PAT | Token with `write:packages` for GHCR push — store as `export GITHUB_TOKEN=...` **never paste into history** |
| kubectl | Configured for your cluster (`cosmic-learning` namespace) |

**Security:** If a token ever appears in `history`, revoke it immediately on GitHub → Settings → Developer settings → Personal access tokens, then create a new one.

---

## Part A — One-time / occasional setup

### A1. Bundle Cobaya data into the image (optional, ~2 GB)

Skip this if you are happy with Pantheon+ only (Dockerfile downloads it during build).

```bash
cd /home/rebecca/CLASS/GA_Project-main

bash docker/prepare_data.sh /home/rebecca/Project_hiclass/cobaya_packages copy

# Sanity check:
ls docker/data/cobaya_packages/code
ls docker/data/cobaya_packages/data/sn_data
```

Alternative (tarball instead of copy):

```bash
bash docker/prepare_data.sh /home/rebecca/Project_hiclass/cobaya_packages tar
```

---

### A2. Publish code to GitHub (when you want a clean public repo)

The export script strips run outputs (`gen_*`, logs, large data).

```bash
cd /home/rebecca/CLASS/GA_Project-main

bash docker/prepare_github_repo.sh /home/rebecca/CLASS/ga-wdm-github

cd /home/rebecca/CLASS/ga-wdm-github
git init
git add .
git commit -m "GA w(a) discovery pipeline with Docker GPU support"
git branch -M main
git remote add origin https://github.com/Rebecca-Briffa/GA_wcdm.git
git push -u origin main
```

If `git remote add` fails because origin already exists:

```bash
git remote set-url origin https://github.com/Rebecca-Briffa/GA_wcdm.git
git push -u origin main
```

**Not pushed to GitHub (by design):** `gen_*` dirs, `docker/data/cobaya_packages/`, build artefacts.

---

## Part B — Build the Docker image (every code or Dockerfile change)

```bash
cd /home/rebecca/CLASS/GA_Project-main

docker build -f docker/Dockerfile -t ga-wdm:latest .
```

Expect **~20–40 min** first time; faster if layers are cached. Image size ~**9.6 GB**.

Verify the image exists:

```bash
docker images ga-wdm:latest
```

---

## Part C — Test locally (laptop / WSL)

### C1. Smoke test — imports and GPU probe

**On WSL without NVIDIA Container Toolkit** (no `--gpus all`):

```bash
docker run --rm -e GA_CLIENT_GPU=cpu ga-wdm:latest check-gpu
```

**On a machine with GPU + nvidia-container-toolkit:**

```bash
docker run --rm --gpus all ga-wdm:latest check-gpu
```

You should see Cobaya/TensorFlow import OK. GPU line shows devices only when GPU passthrough works.

---

### C2. Short GA run (writes to a host folder)

```bash
mkdir -p /home/rebecca/CLASS/ga-docker-test

# WSL / no GPU:
docker run --rm \
  -e GA_CLIENT_GPU=cpu \
  -e CLIENT_PROFILE=ga_fast \
  -v /home/rebecca/CLASS/ga-docker-test:/work \
  ga-wdm:latest run-ga

# With GPU:
docker run --rm --gpus all \
  --env-file docker/env.example \
  -v /home/rebecca/CLASS/ga-docker-test:/work \
  ga-wdm:latest run-ga
```

**Check outputs on the host:**

```bash
ls /home/rebecca/CLASS/ga-docker-test/
tail -f /home/rebecca/CLASS/ga-docker-test/GA_run.log
```

A successful run creates `GA_run.log`, `GA_summary.txt`, and `gen_0_model_*` directories.

**Interactive shell** (debugging):

```bash
docker run --rm -it \
  -e GA_CLIENT_GPU=cpu \
  -v /home/rebecca/CLASS/ga-docker-test:/work \
  ga-wdm:latest bash
# inside: check-gpu && run-ga
```

---

## Part D — Push image to GitHub Container Registry (GHCR)

```bash
export GITHUB_TOKEN='your-token-here'   # use a fresh token, not in history

echo "$GITHUB_TOKEN" | docker login ghcr.io -u rebecca-briffa --password-stdin

docker tag ga-wdm:latest ghcr.io/rebecca-briffa/ga-wdm:latest
docker push ghcr.io/rebecca-briffa/ga-wdm:latest
```

Verify on GitHub: **Your profile → Packages → `ga-wdm`**.

---

## Part E — Deploy and test on Kubernetes (`cosmic-learning`)

### E1. Configure runtime choices

Edit **`docker/k8s/ga-config.yaml`** (likelihood, GPU profile, output path). Example defaults:

- `GA_PROJECT_ROOT` / `GA_OUTPUT_DIR` → `/mnt/ga-run` (PVC)
- `GA_LIKELIHOOD` → `client`
- `CLIENT_PROFILE` → `ga_fast_gpu`
- `GA_CLIENT_GPU` → `auto`

Apply config and pod:

```bash
cd /home/rebecca/CLASS/GA_Project-main

kubectl apply -f docker/k8s/ga-config.yaml
kubectl delete pod -n cosmic-learning ga-wdm-pod --ignore-not-found
kubectl apply -f docker/k8s/ga-pod-interactive.yaml

kubectl get pod -n cosmic-learning ga-wdm-pod
kubectl describe pod -n cosmic-learning ga-wdm-pod   # wait until Running
```

If the pod was already running and you only changed the **image**, delete the pod so it pulls the new tag (`imagePullPolicy: Always`).

---

### E2. Test inside the cluster

**GPU check:**

```bash
kubectl exec -n cosmic-learning ga-wdm-pod -- check-gpu
```

**Shell in:**

```bash
kubectl exec -it -n cosmic-learning ga-wdm-pod -- bash
```

Inside the pod:

```bash
check-gpu
mkdir -p /mnt/ga-run
run-ga
```

**Or one-shot from laptop:**

```bash
kubectl exec -n cosmic-learning ga-wdm-pod -- run-ga
```

**Watch logs:**

```bash
kubectl logs -n cosmic-learning ga-wdm-pod -f
kubectl exec -n cosmic-learning ga-wdm-pod -- tail -f /mnt/ga-run/GA_run.log
```

**Fetch a result file:**

```bash
kubectl cp cosmic-learning/ga-wdm-pod:/mnt/ga-run/GA_run.log ./GA_run.log
```

---

### E3. One-shot Job (runs GA and exits)

```bash
kubectl apply -f docker/k8s/ga-config.yaml
kubectl delete job -n cosmic-learning ga-wdm-run --ignore-not-found
kubectl apply -f docker/k8s/ga-job-run.yaml
kubectl logs -n cosmic-learning -f job/ga-wdm-run
```

---

## Quick reference — full “next time” checklist

```text
[ ] cd GA_Project-main
[ ] (optional) bash docker/prepare_data.sh ... copy
[ ] docker build -f docker/Dockerfile -t ga-wdm:latest .
[ ] docker run ... check-gpu          # local smoke test
[ ] docker run ... run-ga             # optional local GA test
[ ] export GITHUB_TOKEN=...
[ ] echo "$GITHUB_TOKEN" | docker login ghcr.io -u rebecca-briffa --password-stdin
[ ] docker tag + docker push ghcr.io/rebecca-briffa/ga-wdm:latest
[ ] kubectl apply -f docker/k8s/ga-config.yaml
[ ] kubectl delete pod ... && kubectl apply -f docker/k8s/ga-pod-interactive.yaml
[ ] kubectl exec ... check-gpu && run-ga
```

---

## Troubleshooting

| Problem | What to do |
|---------|------------|
| `docker run --gpus all` fails on WSL | Use `-e GA_CLIENT_GPU=cpu`; real GPU testing on cluster |
| `docker login` fails | Use `--password-stdin`, not bare pipe without it |
| `ImagePullBackOff` on cluster | Re-push image; check `ghcr-secret` in namespace |
| Build re-downloads Cobaya | Run `prepare_data.sh copy` so `docker/data/cobaya_packages` exists |
| GA slow / OOM on cluster | Raise memory in `ga-pod-interactive.yaml`; outputs go to PVC `/mnt/ga-run` |
| Change likelihood or profile | Edit `ga-config.yaml`, re-apply, recreate pod |

---

## Related docs

- `docker/CLUSTER.md` — HPC / Slurm / env variables
- `docker/k8s/README.md` — Kubernetes details
- `docker/env.example` — all runtime env vars for `docker run --env-file`
