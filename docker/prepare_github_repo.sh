#!/usr/bin/env bash
# Create a clean directory suitable for GitHub (no gen_* run outputs).
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-${SRC}/../ga-wdm-github}"

echo "Exporting clean repo: ${SRC} → ${DEST}"
rm -rf "${DEST}"
mkdir -p "${DEST}"

rsync -a \
  --exclude 'gen_*' \
  --exclude 'test_w*' \
  --exclude 't1/' \
  --exclude '*.log' \
  --exclude 'GA_summary.txt' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'class/python/build' \
  --exclude 'class/python/classy.cpython*.so' \
  --exclude 'docker/data/cobaya_packages' \
  --exclude 'docker/data/*.tar.gz' \
  "${SRC}/" "${DEST}/"

echo ""
echo "Done. Next steps:"
echo "  cd ${DEST}"
echo "  git init && git add . && git commit -m 'Initial GA w(a) discovery pipeline'"
echo "  git remote add origin https://github.com/YOURUSER/ga-wdm.git"
echo "  git push -u origin main"
echo ""
echo "Build container:"
echo "  cd ${DEST} && docker build -f docker/Dockerfile -t ga-wdm:latest ."
