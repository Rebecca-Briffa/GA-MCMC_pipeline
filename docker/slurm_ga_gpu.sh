#!/bin/bash
#SBATCH --job-name=ga-wdm
#SBATCH --output=ga-wdm-%j.out
#SBATCH --error=ga-wdm-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Example Slurm job — edit partition/account/modules for your cluster.
#
# Submit:
#   sbatch docker/slurm_ga_gpu.sh
#
# Or interactively:
#   srun --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash
#   cd /path/to/ga-wdm && ./docker/run_cluster.sh

set -euo pipefail

# ---- cluster-specific (EDIT THESE) ----
export GA_IMAGE="${GA_IMAGE:-ga-wdm:latest}"
export GA_WORK="${GA_WORK:-${SCRATCH:-/tmp}/ga-wdm-${SLURM_JOB_ID:-local}}"
# ---------------------------------------

mkdir -p "${GA_WORK}"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Work directory: ${GA_WORK}"
echo "Image: ${GA_IMAGE}"

# Load NVIDIA runtime if your site uses modules
# module load singularity  # or: module load docker

if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || true
fi

# ---- Docker (NVIDIA Container Toolkit) ----
if command -v docker &>/dev/null; then
  docker run --rm \
    --gpus all \
    -e GA_PROJECT_ROOT=/work \
    -e GA_OUTPUT_DIR=/work \
    -e GA_LIKELIHOOD=client \
    -e GA_COBAYA_CONFIG=sn_desi \
    -e CLIENT_PROFILE=ga_converge \
    -e GA_CLIENT_GPU=auto \
    -e COBAYA_NOMPI=1 \
    -v "${GA_WORK}:/work" \
    "${GA_IMAGE}" \
    run-ga
  exit 0
fi

# ---- Singularity / Apptainer (common on HPC) ----
if command -v singularity &>/dev/null; then
  SIF="${GA_IMAGE%.latest}.sif"
  singularity exec --nv \
    -B "${GA_WORK}:/work" \
    --env GA_PROJECT_ROOT=/work \
    --env GA_OUTPUT_DIR=/work \
    --env GA_LIKELIHOOD=client \
    --env GA_COBAYA_CONFIG=sn_desi \
    --env CLIENT_PROFILE=ga_converge \
    --env GA_CLIENT_GPU=auto \
    --env COBAYA_NOMPI=1 \
    "${SIF}" \
    run-ga
  exit 0
fi

echo "Neither docker nor singularity found." >&2
exit 1
