#!/usr/bin/env bash
# Install Cobaya likelihood data into COBAYA_PACKAGES_PATH.
# Called at Docker build time and optionally at container start.
set -euo pipefail

PACKAGES="${COBAYA_PACKAGES_PATH:-/opt/cobaya_packages}"
MODE="${COBAYA_DATA_MODE:-pantheon}"

mkdir -p "${PACKAGES}"
export COBAYA_PACKAGES_PATH="${PACKAGES}"

echo "=== Cobaya data install (mode=${MODE}) → ${PACKAGES} ==="

case "${MODE}" in
  pantheon)
    cobaya-install sn.pantheonplusshoes
    ;;
  sn_desi|desi)
    # Pantheon+SH0ES + DESI DR2 BAO (default GA HPC config)
    cobaya-install sn.pantheonplusshoes
    cobaya-install bao.desi_dr2
    ;;
  minimal)
    cobaya-install sn.pantheonplusshoes
    ;;
  full)
    cobaya-install sn.pantheonplusshoes
    cobaya-install bao.desi_dr2
    cobaya-install planck_2018_lowl.TT planck_2018_lowl.EE planck_NPIPE_highl_CamSpec.TTTEEE
    ;;
  skip)
    echo "Skipping cobaya-install (expect data at ${PACKAGES})"
    ;;
  *)
    echo "Unknown COBAYA_DATA_MODE=${MODE}" >&2
    exit 1
    ;;
esac

echo "=== Cobaya packages installed ==="
ls -la "${PACKAGES}" || true
