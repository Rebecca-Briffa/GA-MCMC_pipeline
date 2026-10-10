#!/usr/bin/env bash
# Bundle Cobaya packages from your local machine for offline Docker/cluster builds.
set -euo pipefail

SRC="${1:-/home/rebecca/Project_hiclass/cobaya_packages}"
DEST_DIR="$(cd "$(dirname "$0")" && pwd)/data"
MODE="${2:-copy}"

if [[ ! -d "${SRC}" ]]; then
  echo "Source not found: ${SRC}" >&2
  echo "Usage: $0 [/path/to/cobaya_packages] [copy|tar]" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"

case "${MODE}" in
  copy)
    echo "Copying ${SRC} → ${DEST_DIR}/cobaya_packages (this may take a while)..."
    rsync -a --delete "${SRC}/" "${DEST_DIR}/cobaya_packages/"
    du -sh "${DEST_DIR}/cobaya_packages"
    ;;
  tar)
    OUT="${DEST_DIR}/cobaya_packages.tar.gz"
    echo "Creating ${OUT} ..."
    tar -czf "${OUT}" -C "$(dirname "${SRC}")" "$(basename "${SRC}")"
    ls -lh "${OUT}"
    ;;
  *)
    echo "Mode must be copy or tar" >&2
    exit 1
    ;;
esac

echo "Ready for: docker build -f docker/Dockerfile -t ga-wdm:latest ."
