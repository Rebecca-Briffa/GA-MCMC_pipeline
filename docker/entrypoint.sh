#!/usr/bin/env bash
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

mkdir -p "${GA_PROJECT_ROOT}" "${GA_OUTPUT_DIR}"

# If packages dir is empty (volume mount), install Pantheon+ on first start
if [[ "${COBAYA_INSTALL_ON_START:-0}" == "1" ]]; then
  if [[ ! -d "${COBAYA_PACKAGES_PATH}/data/sn_data" ]]; then
    echo "Installing Cobaya data on container start..."
    bash /opt/ga-project/docker/install_cobaya_data.sh
  fi
fi

cd /opt/ga-project/class/scripts

if [[ "${1:-}" == "bash" || "${1:-}" == "sh" ]]; then
  exec "$@"
fi

if [[ "${1:-}" == "check-gpu" ]]; then
  exec python check_client_gpu.py
fi

if [[ "${1:-}" == "run-ga" ]]; then
  shift
  exec bash /opt/ga-project/docker/run_ga.sh "$@"
fi

exec "$@"
