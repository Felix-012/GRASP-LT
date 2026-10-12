#!/usr/bin/env bash
# Shared driver for the run files. Usage: runs/_run.sh <stage|all> <method> [<method> ...]
#   PROFILE   paper (default) | single_gpu | reduced | smoke        (reproduce/profiles.py)
#   LAUNCHER  local (default) | slurm
#   SEEDS     space-separated generator seeds (default: the paper seeds of each method)
#   EXTRA     extra arguments for reproduce.cli (e.g. "--fix_gen_embeddings" or "--dry_run")
set -euo pipefail
cd "$(dirname "$0")/.."
stage="$1"; shift
PROFILE="${PROFILE:-paper}"; LAUNCHER="${LAUNCHER:-local}"
seed_args=(); for s in ${SEEDS:-}; do seed_args+=(--seed "$s"); done
for m in "$@"; do
  echo ">>> $m ($stage, profile=$PROFILE, launcher=$LAUNCHER)"
  if [[ "$m" == real ]]; then
    # the real-data reference only has a classify stage
    if [[ "$stage" != all && "$stage" != classify && "$stage" != eval ]]; then echo "    (skipped: stage $stage)"; continue; fi
    python -m reproduce.cli run --method real --stage classify --profile "$PROFILE" --launcher "$LAUNCHER" ${EXTRA:-}
  else
    rc=0
    python -m reproduce.cli run --method "$m" --stage "$stage" --profile "$PROFILE" --launcher "$LAUNCHER" \
      "${seed_args[@]}" ${EXTRA:-} || rc=$?
    if [[ $rc -eq 3 ]]; then echo "    (skipped: not runnable with PROFILE=$PROFILE)"; elif [[ $rc -ne 0 ]]; then exit $rc; fi
  fi
done
