# Cobaya configs for the GA

Choose a config at the top of `GA_MCMC_new.py`:

```python
COBAYA_CONFIG = "cmb_spa"   # or "sn_desi", or a full path to a .yaml
```

Or via environment:

```bash
export GA_COBAYA_CONFIG=cmb_spa
# or absolute path:
export GA_COBAYA_YAML=/path/to/custom.yaml
```

Available stems in this folder are listed at launch.

## Configs

| Name | File | Contents |
|------|------|----------|
| `cmb_spa` | `cmb_spa.yaml` | DESI + Pantheon+SH0ES + Planck + ACT + SPT + MUSE |
| `sn_desi` | `sn_desi.yaml` | Pantheon+SH0ES + DESI DR2 (fast SN+BAO) |

Add a new choice by dropping `my_data.yaml` here and setting `COBAYA_CONFIG = "my_data"`.

### One-time package install (CMB-SPA)

```bash
export COBAYA_PACKAGES_PATH=/path/to/cobaya_packages
export COBAYA_NOMPI=1
cobaya-install cobaya_configs/cmb_spa.yaml -p "$COBAYA_PACKAGES_PATH" --no-set-global

# SPT needs the real CMB candl (NOT the PyPI neural-net package named candl):
pip install "git+https://github.com/Lbalkenhol/candl.git"
```

### Optional GA speed cap

```bash
export GA_COBAYA_MAX_SAMPLES=500
```
