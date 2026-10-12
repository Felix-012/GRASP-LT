#!/usr/bin/env bash
# Tables 1-2, +MG block (generation only; needs the ft_class / grasp_* checkpoints of table1-3_mimic.sh).
# Usage: [PROFILE=paper|single_gpu|reduced|smoke] [LAUNCHER=local|slurm] [STAGE=all|train|generate|metrics|classify] runs/table1-2_minority_guidance.sh
set -euo pipefail
exec "$(dirname "$0")/_run.sh" "${STAGE:-all}" vanilla_mg grasp_lowdim_mg grasp_highdim_mg
