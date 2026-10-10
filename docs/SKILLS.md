# Skills demonstrated

Employer-facing summary of what this repository shows in practice.

## Scientific computing

- Designing a **search space** for dynamical dark energy \(w(a)\) (basis shapes, complexity caps, identifiability via unique-shape repair).
- Coupling a Boltzmann code (**CLASS**) to Bayesian likelihoods (**Cobaya**) and an emulator (**CLiENT** / TensorFlow).
- Using **information criteria** (DIC vs BIC/AIC) and **posterior diagnostics** as selection objectives, not just maximum likelihood.

## Software engineering

- Modular Python pipeline: encoding → theory injection → parallel evaluation → summary I/O.
- Configuration via environment / ConfigMap (12-factor style), not hardcoded cluster paths in source.
- Reproducible **container** build with a thin entrypoint (`run-ga`) and documented data prep.

## HPC / MLOps

- Multi-GPU parallel fitness (`GA_PARALLEL_WORKERS`, `GA_GPU_IDS`).
- Kubernetes manifests (ConfigMap, interactive pod, Job) and Slurm helpers.
- Operational hardening: output directory creation, population-size alignment, failure metadata in summaries.

## Communication

- Outer README + focused docs for pipeline, encoding, fitness, and deployment.
- Curated figures and summaries suitable for talks and paper drafts without shipping multi-GB run trees.
