#!/usr/bin/env bash
# Tables 1-3 (MIMIC-CXR-LT): all methods and the real-data classifier. Exact reproduction of the paper rows.
# Usage: [PROFILE=paper|single_gpu|reduced|smoke] [LAUNCHER=local|slurm] [STAGE=all|train|generate|metrics|classify] runs/table1-3_mimic.sh
set -euo pipefail
exec "$(dirname "$0")/_run.sh" "${STAGE:-all}" ft_class ft_text_class mocle ccua adapter_768 grasp_768 grasp_lowdim grasp_highdim moe_grasp_lowdim moe_grasp_highdim real
