#!/usr/bin/env bash
# ImageNet-LT inputs (Tab. 2, right block) under $GRASP_DATA/imagenet-lt.
#   IMAGENET_LT_TXT  folder with ImageNet_LT_train.txt / _val.txt / _test.txt (Liu et al., 2019)
#   IMAGENET_CLSIDX  imagenet1000_clsidx_to_labels.txt (class index -> name, a Python dict literal)
#   IMAGENET_ROOT    ImageNet-2012 image root
# Prompts are the class names (the "Finding Labels" column written by custom_datasets/add_imagenet_labels.py).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GRASP_DATA:?}" "${IMAGENET_ROOT:?}"
I="$GRASP_DATA/imagenet-lt"; mkdir -p "$I/text_cluster/train"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-dev}"
if [[ ! -f "$I/imagenet_lt_train.csv" || ! -f "$I/imagenet_lt_test.csv" ]]; then
  : "${IMAGENET_LT_TXT:?set IMAGENET_LT_TXT (official split files)}" "${IMAGENET_CLSIDX:?set IMAGENET_CLSIDX}"
  python custom_datasets/make_imagenet_lt_csv.py --txt_dir "$IMAGENET_LT_TXT" --clsidx "$IMAGENET_CLSIDX" --out_dir "$I"
fi
[[ -f "$I/imagenet_train_latents_flux-lt.h5" ]] || python custom_datasets/precompute_latents.py \
    --csv "$I/imagenet_lt_train.csv" --image_root "$IMAGENET_ROOT" --out "$I/imagenet_train_latents_flux-lt.h5"
for split in train test; do
  [[ -f "$I/embeddings_imagenet-lt_flux_${split}.h5" ]] || python custom_datasets/precompute_embeddings.py \
      --csv-path "$I/imagenet_lt_${split}.csv" --text-column "Finding Labels" \
      --output-h5-path "$I/embeddings_imagenet-lt_flux_${split}.h5" --dtype float32
  [[ -f "$I/imagenet-lt_${split}.h5" ]] || python utils/make_eval_h5.py --csv "$I/imagenet_lt_${split}.csv" \
      --root "$IMAGENET_ROOT" --label_col "Finding Labels" --out "$I/imagenet-lt_${split}.h5" --mode raw
done
[[ -f "$I/text_cluster/train/centroids.npz" ]] || python partitions/text_clusters.py \
    --train_embeddings "$I/embeddings_imagenet-lt_flux_train.h5" --out_dir "$I/text_cluster/train" --K 4
