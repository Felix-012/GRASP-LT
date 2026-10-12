#!/usr/bin/env bash
# Downstream evaluation set: NIH-CXR-LT test images with one of the 13 labels shared with MIMIC-CXR-LT.
#   LONGTAIL_DIR  LongTailCXR labels/ folder;  NIH_IMAGES  ChestX-ray14 images/ folder
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GRASP_DATA:?}" "${LONGTAIL_DIR:?}" "${NIH_IMAGES:?}"
N="$GRASP_DATA/nih-cxr-lt"; mkdir -p "$N"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/make_nih_eval_csv.py --nih_csv "$LONGTAIL_DIR/nih-cxr-lt_single-label_test.csv" \
    --image_root "$NIH_IMAGES" --out "$N/cxr14-lt_test.csv"
[[ -f "$N/classifier_eval.h5" ]] || python utils/make_eval_h5.py --csv "$N/cxr14-lt_test.csv" \
    --out "$N/classifier_eval.h5" --mode raw
