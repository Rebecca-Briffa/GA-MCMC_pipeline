#!/usr/bin/env bash
# Quick launcher on a cluster login node (Docker + GPU).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${GA_IMAGE:-ga-wdm:latest}"
WORK="${GA_WORK:-${SCRATCH:-$HOME}/ga-wdm-runs}"

mkdir -p "${WORK}"

echo "Repository : ${REPO}"
echo "Image      : ${IMAGE}"
echo "Output dir : ${WORK}  (gen_* models + GA_run.log)"

docker run --rm -it \
  --gpus all \
  -v "${WORK}:/work" \
  -e GA_PROJECT_ROOT=/work \
  -e GA_OUTPUT_DIR=/work \
  -e GA_LIKELIHOOD="${GA_LIKELIHOOD:-client}" \
  -e GA_COBAYA_CONFIG="${GA_COBAYA_CONFIG:-sn_desi}" \
  -e CLIENT_PROFILE="${CLIENT_PROFILE:-ga_converge}" \
  -e GA_CLIENT_GPU="${GA_CLIENT_GPU:-auto}" \
  -e COBAYA_NOMPI=1 \
  "${IMAGE}" \
  "${@:-run-ga}"
