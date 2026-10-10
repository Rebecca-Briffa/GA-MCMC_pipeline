#!/usr/bin/env bash
# Launch standalone fluid ΛCDM CLiENT (SN+DESI) on the cosmic-learning pod.
#
# Run this from a machine with kubectl (e.g. mars), from the GA_Github_0708 tree:
#
#   bash lcdm_baseline/launch_on_pod.sh
#
# Requires: updated params.py (is_lcdm_expr / LCDM branch) on the pod — this
# script copies it together with the runner.

set -euo pipefail

NS="${GA_K8S_NS:-cosmic-learning}"
POD="${GA_K8S_POD:-ga-wdm-pod}"
OUT="${GA_LCDM_OUT:-/mnt/lcdm-baseline}"
PROFILE="${CLIENT_PROFILE:-ga_converge}"
CONFIG="${GA_COBAYA_CONFIG:-sn_desi}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Copy runner + params.py into pod:${OUT} ==="
kubectl exec -n "$NS" "$POD" -- mkdir -p "$OUT"
kubectl cp "$ROOT/lcdm_baseline/run_lcdm_client.py" "$NS/$POD:$OUT/run_lcdm_client.py"
# Ensure LCDM sentinel is available even if the image is older:
kubectl cp "$ROOT/class/scripts/params.py" "$NS/$POD:/opt/ga-project/class/scripts/params.py"

echo "=== Start CLiENT ΛCDM in background (profile=${PROFILE}) ==="
kubectl exec -n "$NS" "$POD" -- bash -lc "
  set -euo pipefail
  export GA_PROJECT_ROOT=${OUT}
  export GA_OUTPUT_DIR=${OUT}
  export GA_COBAYA_CONFIG=${CONFIG}
  export CLIENT_PROFILE=${PROFILE}
  export GA_LIKELIHOOD=client
  export COBAYA_NOMPI=1
  export GA_CLIENT_GPU=auto
  cd /opt/ga-project/class/scripts
  nohup python ${OUT}/run_lcdm_client.py --rebuild \
    > ${OUT}/lcdm_run.log 2>&1 &
  echo \"PID=\$!\"
  echo \"log: ${OUT}/lcdm_run.log\"
"

echo
echo "Watch progress:"
echo "  kubectl exec -n $NS $POD -- tail -f $OUT/lcdm_run.log"
echo "When finished:"
echo "  kubectl exec -n $NS $POD -- cat $OUT/lcdm_summary.txt"
echo "  kubectl cp $NS/$POD:$OUT/lcdm_summary.txt ./lcdm_summary.txt"
