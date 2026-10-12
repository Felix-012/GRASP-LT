# Data

All inputs live under one data root, `$GRASP_DATA`. No patient data and nothing derived from report text is
distributed with this repository; you build everything from the original sources with the scripts below.
`python tools/verify_data.py` compares your files with the checksums of the files used for the paper
(`data/checksums.tsv`).

## Sources

| dataset | where | access |
|---|---|---|
| MIMIC-CXR-JPG v2 (images) and MIMIC-CXR v2 (reports) | physionet.org | credentialed PhysioNet account, data use agreement |
| MIMIC-CXR-LT / NIH-CXR-LT single-label splits | github.com/VITA-Group/LongTailCXR (`labels/`) | public |
| NIH ChestX-ray14 images | nihcc.app.box.com/v/ChestXray-NIHCC | public |
| ImageNet-2012 + ImageNet-LT split files (Liu et al., 2019) | image-net.org, github.com/zhmiao/OpenLongTailRecognition-OLTR | ImageNet terms |
| FLUX.1-dev | huggingface.co/black-forest-labs/FLUX.1-dev | gated, non-commercial license |
| CXR feature extractor for the metrics (DenseNet-121 trained on ChestX-ray14, `m-05122024-131940.pth.tar`) | TODO: download link | set `GRASP_CXR_MODEL` to the file |

## Layout

```
$GRASP_DATA/
  mimic-lt/
    mimic-cxr-lt_train.csv, mimic-cxr-lt_test.csv     87,493 / 23,550 rows; columns incl. path, Finding Labels, impression
    image_latents_flux-lt.h5                         FLUX VAE latent mean/logvar per training image (item_<i>)
    embeddings_flux-lt.h5, embeddings_flux-lt_test.h5                       impression prompts (GRASP, MoCLE, MoE-GRASP)
    embeddings_mimic-lt_labels_train.h5, embeddings_mimic-lt_labels_test.h5 finding-label prompts (FT (class), CCUA)
    embeddings_mimic-lt_label_txt_train.h5                                  label + text prompts (FT (text+class))
    null_embedding.h5                                 empty-prompt embedding (CCUA)
    labels_gt.txt, test/labels_gt.txt                 GRASP routing files (expert id per row)
    (Table 4 partition variants: build with partitions/build_label_files.py --map <variant>; not needed for Tables 1-3)
    text_cluster/train/labels.txt, centroids.npz      text-embedding partition (Embed, MoCLE)
    mimic-lt_train.h5, mimic-lt_test.h5               real images for the metrics and the real-data classifier
  nih-cxr-lt/
    cxr14-lt_test.csv, classifier_eval.h5             16,506 NIH test images with one of the 13 shared labels
  imagenet-lt/
    imagenet_lt_train.csv, imagenet_lt_test.csv, imagenet_train_latents_flux-lt.h5,
    embeddings_imagenet-lt_flux_{train,test}.h5, imagenet-lt_{train,test}.h5, text_cluster/train/...
```

## Building it

```bash
export GRASP_DATA=/path/to/data
export LONGTAIL_DIR=/path/to/LongTailCXR/labels
export MIMIC_JPG_ROOT=/path/to/mimic-cxr-jpg/2.0.0 MIMIC_REPORTS_ROOT=/path/to/mimic-cxr/2.0.0   # both contain files/
export NIH_IMAGES=/path/to/ChestXray14/images
scripts/prepare_mimic.sh        # CSVs, latents, embeddings, routing files, text clusters, real-image HDF5s
scripts/prepare_nih.sh          # downstream evaluation set
IMAGENET_LT_TXT=... IMAGENET_CLSIDX=... IMAGENET_ROOT=... scripts/prepare_imagenet.sh   # ImageNet-LT block of Table 2
python tools/verify_data.py
```

Notes on individual steps:

* **CSVs.** `utils/add_info.py` turns the one-hot LongTailCXR columns into a `Finding Labels` column (and
  merges impressions if you already have them, `MIMIC_IMPRESSIONS_CSV`); `utils/add_impression.py` fills
  the `impression` column from the report text (IMPRESSION section, falling back to FINDINGS, then to the
  label name). The impression is the generation prompt for GRASP, MoCLE and MoE-GRASP.
* **Embeddings** are stored in float32 with 256 T5 tokens (`custom_datasets/precompute_embeddings.py
  --dtype float32`). The `label_txt` file of FT (text+class) encodes the finding label together with the
  report text.
* **Routing files** are written from the exact label-to-expert maps in `partitions/maps.json`
  (`partitions/build_label_files.py`) and are byte-identical to the paper files when the CSV row order
  matches (checked by `tools/verify_data.py`). See [PARTITIONS.md](PARTITIONS.md).
* **Real-image HDF5s** (`utils/make_eval_h5.py`) store encoded images plus `targets`, `classes` and `paths`.
  The real-data reference classifier of Table 3 is trained on `mimic-lt_test.h5`, i.e. on the real images
  of the same 23,550 studies whose prompts are used for generation.
* **NIH evaluation set.** `tools/make_nih_eval_csv.py` keeps the NIH-CXR-LT test images whose label is one
  of the 13 labels shared with MIMIC-CXR-LT (NIH `Effusion` = MIMIC `Pleural Effusion`): 16,506 of 20,279.

## Tiny subset

`tools/make_tiny_subset.py --src $GRASP_DATA --dst /tmp/grasp_tiny` writes a consistent subset (default
256 train / 128 test / 256 NIH rows, every finding included) for smoke tests and development.

ImageNet-LT CSVs: `custom_datasets/make_imagenet_lt_csv.py` builds `imagenet_lt_{train,test}.csv` from the official
split files (`ImageNet_LT_{train,val,test}.txt`) and the class-name list; it reproduces the paper files byte for
byte when the split files are in their published order (checked against the paper CSVs). The test-row order fixes
which noise each prompt receives, so run `tools/verify_data.py` afterwards: both CSVs are listed in
`data/checksums.tsv`.

