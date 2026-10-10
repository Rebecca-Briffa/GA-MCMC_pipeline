# GA + CLASS + Cobaya + CLiENT pipeline for HPC / GPU clusters
#
# Build (from GA_Project-main repo root):
#   docker build -f docker/Dockerfile -t ga-wdm:latest .
#
# With pre-downloaded Cobaya packages (optional, ~2GB):
#   tar -xzf docker/data/cobaya_packages.tar.gz -C docker/data/
#   docker build -f docker/Dockerfile -t ga-wdm:latest .
#
# Run with GPU:
#   docker run --gpus all -v $PWD/runs:/work ga-wdm:latest run-ga

FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ---------------------------------------------------------------------------
# 1. System build dependencies (CLASS C code + Python)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git gfortran \
    python3.10 python3.10-dev python3-pip \
    libgsl-dev libopenblas-dev liblapack-dev \
    wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

RUN python3 -m pip install --upgrade pip wheel setuptools

# ---------------------------------------------------------------------------
# 2. Python stack (GA + Cobaya + CLiENT / TensorFlow GPU)
# ---------------------------------------------------------------------------
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# 3. CLiENT (likelihood emulator)
# ---------------------------------------------------------------------------
WORKDIR /opt
RUN git clone --depth 1 https://github.com/AarhusCosmology/client_public.git client_public

# ---------------------------------------------------------------------------
# 4. CLASS template (kept as source — GA copies this per model)
# ---------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/lesgourg/class_public.git class_src
WORKDIR /opt/class_src
RUN make clean && make -j"$(nproc)" class && make -j"$(nproc)" libclass.a

# ---------------------------------------------------------------------------
# 5. GA project (scripts + CLASS template tree)
# ---------------------------------------------------------------------------
WORKDIR /opt/ga-project

# Copy pipeline code; .dockerignore excludes gen_* run artefacts
COPY class ./class
COPY docs ./docs
COPY docker ./docker

# Remove stale build artefacts from template if present
RUN find ./class -type d -name build -prune -exec rm -rf {} + 2>/dev/null || true \
    && find ./class -name "*.so" -delete 2>/dev/null || true \
    && rm -rf ./class/python/build ./class/python/classy.cpython*.so 2>/dev/null || true

# Verify CLASS template compiles inside the image
WORKDIR /opt/ga-project/class
RUN make clean && make -j"$(nproc)" class && make -j"$(nproc)" libclass.a

# ---------------------------------------------------------------------------
# 6. Cobaya likelihood data
# ---------------------------------------------------------------------------
ENV COBAYA_PACKAGES_PATH=/opt/cobaya_packages
# sn_desi = Pantheon+SH0ES + DESI DR2 (matches GA_COBAYA_CONFIG default)
ENV COBAYA_DATA_MODE=sn_desi

# Option A: copy pre-staged packages from build context (if present)
COPY docker/data/ /tmp/cobaya_data/
RUN if [ -d /tmp/cobaya_data/cobaya_packages ]; then \
      mv /tmp/cobaya_data/cobaya_packages /opt/cobaya_packages; \
      echo "Using bundled cobaya_packages from build context"; \
    elif [ -f /tmp/cobaya_data/cobaya_packages.tar.gz ]; then \
      mkdir -p /opt && tar -xzf /tmp/cobaya_data/cobaya_packages.tar.gz -C /opt; \
      echo "Extracted cobaya_packages.tar.gz"; \
    else \
      mkdir -p /opt/cobaya_packages && \
      COBAYA_PACKAGES_PATH=/opt/cobaya_packages COBAYA_DATA_MODE=sn_desi \
        bash /opt/ga-project/docker/install_cobaya_data.sh; \
    fi \
    && rm -rf /tmp/cobaya_data

# Runtime choices (GA_LIKELIHOOD, CLIENT_PROFILE, GPU, output paths) are NOT set here.
# Defaults live in docker/entrypoint.sh; override via:
#   - docker/k8s/ga-config.yaml  (Kubernetes ConfigMap)
#   - docker/env.example         (docker run --env-file)
#   - -e VAR=value at docker run / kubectl env

RUN chmod +x /opt/ga-project/docker/*.sh \
    && ln -sf /opt/ga-project/docker/run_ga.sh /usr/local/bin/run-ga \
    && printf '%s\n' '#!/usr/bin/env bash' 'exec python /opt/ga-project/class/scripts/check_client_gpu.py "$@"' \
       > /usr/local/bin/check-gpu && chmod +x /usr/local/bin/check-gpu

WORKDIR /work

# Smoke tests
RUN python -c "import cobaya; print('Cobaya', cobaya.__version__)"
RUN python -c "import pygad, sympy, sklearn, tensorflow as tf; print('TF', tf.__version__, 'GPUs', tf.config.list_physical_devices('GPU'))"

ENTRYPOINT ["/opt/ga-project/docker/entrypoint.sh"]
CMD ["bash"]
