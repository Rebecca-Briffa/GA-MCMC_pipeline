#!/usr/bin/env bash
# Run the GA inside the container (GPU + CLiENT defaults).
set -euo pipefail

export COBAYA_NOMPI="${COBAYA_NOMPI:-1}"
export GA_PROJECT_ROOT="${GA_PROJECT_ROOT:-/work}"
export GA_OUTPUT_DIR="${GA_OUTPUT_DIR:-${GA_PROJECT_ROOT}}"
export GA_CLASS_DIR="${GA_CLASS_DIR:-/opt/ga-project/class}"
export CLIENT_ROOT="${CLIENT_ROOT:-/opt/client_public}"
export COBAYA_PACKAGES_PATH="${COBAYA_PACKAGES_PATH:-/opt/cobaya_packages}"
export GA_LIKELIHOOD="${GA_LIKELIHOOD:-client}"
export GA_COBAYA_CONFIG="${GA_COBAYA_CONFIG:-sn_desi}"
export CLIENT_PROFILE="${CLIENT_PROFILE:-ga_converge}"
export GA_CLIENT_GPU="${GA_CLIENT_GPU:-auto}"
export CLIENT_VALIDATE_MAP="${CLIENT_VALIDATE_MAP:-1}"

mkdir -p "${GA_OUTPUT_DIR}" "${GA_PROJECT_ROOT}"

cd /opt/ga-project/class/scripts

echo "=== GA run configuration ==="
echo "GA_PROJECT_ROOT     = ${GA_PROJECT_ROOT}"
echo "GA_OUTPUT_DIR       = ${GA_OUTPUT_DIR}"
echo "GA_CLASS_DIR        = ${GA_CLASS_DIR}"
echo "CLIENT_ROOT         = ${CLIENT_ROOT}"
echo "COBAYA_PACKAGES_PATH= ${COBAYA_PACKAGES_PATH}"
echo "GA_LIKELIHOOD       = ${GA_LIKELIHOOD}"
echo "GA_COBAYA_CONFIG    = ${GA_COBAYA_CONFIG}"
echo "GA_INITIAL_CATALOG  = ${GA_INITIAL_CATALOG:-canonical_plus_enumerate}"
echo "GA_POP_SIZE         = ${GA_POP_SIZE:-10}"
echo "CLIENT_PROFILE      = ${CLIENT_PROFILE}"
echo "GA_CLIENT_GPU       = ${GA_CLIENT_GPU}"
echo "=============================="

python check_client_gpu.py || true
exec python GA_MCMC_new.py "$@"
