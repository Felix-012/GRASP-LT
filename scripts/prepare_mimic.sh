#!/usr/bin/env bash
# Build every MIMIC-CXR-LT input of the paper under $GRASP_DATA/mimic-lt (see docs/DATA.md).
# Each step is skipped when its output exists. Steps 2-4 need one GPU and FLUX.1-dev access.
#
# Required environment:
#   GRASP_DATA            output data root
#   LONGTAIL_DIR          LongTailCXR label folder (github.com/VITA-Group/LongTailCXR, labels/)
#   MIMIC_JPG_ROOT        MIMIC-CXR-JPG version folder that CONTAINS files/ (e.g. .../mimic-cxr-jpg/2.0.0);
#                         the CSV 'path' column starts with files/ (PhysioNet, credentialed access)
#   MIMIC_REPORTS_ROOT    MIMIC-CXR version folder that CONTAINS files/ with the report .txt files
# Optional:
#   MIMIC_IMPRESSIONS_CSV CSV with dicom_id + impression (if you already extracted impressions)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GRASP_DATA:?}" "${LONGTAIL_DIR:?}" "${MIMIC_JPG_ROOT:?}" "${MIMIC_REPORTS_ROOT:?}"
D="$GRASP_DATA/mimic-lt"; mkdir -p "$D/test" "$D/text_cluster/train"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-dev}"
step() { echo; echo "==== $*"; }

step "1/6 split CSVs with 'Finding Labels' and 'impression' columns"
for split in train test; do
  out="$D/mimic-cxr-lt_${split}.csv"
  [[ -f "$out" ]] && { echo "exists: $out"; continue; }
  python utils/add_info.py --csv-a "$LONGTAIL_DIR/mimic-cxr-lt_single-label_${split}.csv" \
      ${MIMIC_IMPRESSIONS_CSV:+--csv-b "$MIMIC_IMPRESSIONS_CSV"} --out "$out.tmp"
  # fill impressions from the report .txt files (fallback: FINDINGS section, then the label name)
  python utils/add_impression.py "$out.tmp" "$MIMIC_REPORTS_ROOT" -o "$out"
  rm -f "$out.tmp"
done

step "2/6 FLUX VAE latents of the training images (1024 x 1024)"
[[ -f "$D/image_latents_flux-lt.h5" ]] || python custom_datasets/precompute_latents.py \
    --csv "$D/mimic-cxr-lt_train.csv" --image_root "$MIMIC_JPG_ROOT" --out "$D/image_latents_flux-lt.h5"

step "3/6 FLUX text embeddings (float32, 256 tokens)"
emb() { [[ -f "$3" ]] || python custom_datasets/precompute_embeddings.py --csv-path "$1" --text-column "$2" \
          --output-h5-path "$3" --dtype float32; }
emb "$D/mimic-cxr-lt_train.csv" impression        "$D/embeddings_flux-lt.h5"
emb "$D/mimic-cxr-lt_test.csv"  impression        "$D/embeddings_flux-lt_test.h5"
emb "$D/mimic-cxr-lt_train.csv" "Finding Labels"  "$D/embeddings_mimic-lt_labels_train.h5"
emb "$D/mimic-cxr-lt_test.csv"  "Finding Labels"  "$D/embeddings_mimic-lt_labels_test.h5"
# FT (text+class) uses a label+text prompt column; see docs/DATA.md ("label_txt") before building it.
[[ -f "$D/null_embedding.h5" ]] || python custom_datasets/precompute_null_embedding.py --output-h5-path "$D/null_embedding.h5"

step "4/6 routing file: the GRASP partition (exact map, partitions/maps.json)"
# partitions/maps.json also holds the Table 4 sensitivity maps; build one with
#   python partitions/build_label_files.py --map <random0|...|hmt> --csv <split csv> --out <file>
for split in train test; do
  sub=$([[ $split == test ]] && echo "test/" || echo "")
  python partitions/build_label_files.py --map submitted --csv "$D/mimic-cxr-lt_${split}.csv" --out "$D/${sub}labels_gt.txt"
done

step "5/6 text-embedding partition (Embed / MoCLE): bisecting k-means, K=4, cosine"
[[ -f "$D/text_cluster/train/centroids.npz" ]] || python partitions/text_clusters.py \
    --train_embeddings "$D/embeddings_flux-lt.h5" --out_dir "$D/text_cluster/train" --K 4

step "6/6 real-image HDF5s for the metrics (train/test) and the real-data classifier"
for split in train test; do
  [[ -f "$D/mimic-lt_${split}.h5" ]] || python utils/make_eval_h5.py --csv "$D/mimic-cxr-lt_${split}.csv" \
      --root "$MIMIC_JPG_ROOT" --out "$D/mimic-lt_${split}.h5" --mode raw
done
python tools/verify_data.py --data "$GRASP_DATA" || true
echo "done: $D"
