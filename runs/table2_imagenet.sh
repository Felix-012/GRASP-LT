#!/usr/bin/env bash
# Table 2, ImageNet-LT columns. Exact reproduction of the paper rows.
# Usage: [PROFILE=paper|single_gpu|reduced|smoke] [LAUNCHER=local|slurm] [STAGE=all|train|generate|metrics|classify] runs/table2_imagenet.sh
set -euo pipefail
exec "$(dirname "$0")/_run.sh" "${STAGE:-all}" imagenet_ft imagenet_mocle imagenet_adapter_768 imagenet_grasp_768 imagenet_grasp_lowdim imagenet_grasp_highdim imagenet_moe_grasp_lowdim
