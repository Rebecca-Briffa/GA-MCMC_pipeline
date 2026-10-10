# Optional: place your Cobaya packages here before building the image.
#
# Minimal (Pantheon+ only, ~160 MB) — auto-downloaded at build if this dir is empty:
#   nothing to do; Dockerfile runs cobaya-install sn.pantheonplusshoes (+ DESI for sn_desi)
#
# Full local copy (~2 GB) from your machine:
#   rsync -a /home/rebecca/Project_hiclass/cobaya_packages/ ./cobaya_packages/
#   docker build -f docker/Dockerfile -t ga-wdm:latest .
#
# Or tarball:
#   tar -czf cobaya_packages.tar.gz -C /home/rebecca/Project_hiclass cobaya_packages
#   mv cobaya_packages.tar.gz docker/data/
#
# Do NOT commit large datasets to GitHub — use Git LFS, cluster filesystem, or build-time download.
