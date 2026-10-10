# Fluid ΛCDM baseline (SN + DESI)

Standalone CLiENT run with **\(w\equiv -1\)** (no free DE parameters), same likelihoods as the GA (`sn_desi`: Pantheon+SH0ES + DESI DR2).

## What it does

1. Builds a CLASS tree with `FLUID_W_FLD ≡ -1`
2. Cobaya params: `Omega_Lambda=0`, `w0_fld=-1`, `wa_fld=0` (fixed)
3. Runs CLiENT (`ga_converge` by default)
4. Writes `lcdm_summary.txt` / `.json` with χ², \(k\), BIC

Use that BIC as the reference for ΔBIC ranking plots.

## Launch on the HPC pod

From a machine with `kubectl` (e.g. `mars`), in `GA_Github_0708`:

```bash
bash lcdm_baseline/launch_on_pod.sh
```

Optional env overrides:

```bash
CLIENT_PROFILE=ga_fast bash lcdm_baseline/launch_on_pod.sh   # quicker scout
```

Watch / fetch:

```bash
kubectl exec -n cosmic-learning ga-wdm-pod -- tail -f /mnt/lcdm-baseline/lcdm_run.log
kubectl cp cosmic-learning/ga-wdm-pod:/mnt/lcdm-baseline/lcdm_summary.txt ./lcdm_summary.txt
```

## Manual one-liner (already inside the pod)

```bash
export GA_PROJECT_ROOT=/mnt/lcdm-baseline GA_OUTPUT_DIR=/mnt/lcdm-baseline
export GA_COBAYA_CONFIG=sn_desi CLIENT_PROFILE=ga_converge GA_LIKELIHOOD=client
cd /opt/ga-project/class/scripts
python /mnt/lcdm-baseline/run_lcdm_client.py --rebuild
```

Requires the updated `params.py` (LCDM sentinel `-1`) on the image or copied into `/opt/ga-project/class/scripts/`.
