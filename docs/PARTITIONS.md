# Partitions

GRASP routes each training sample to one expert through a fixed lookup π from conditions to experts.
On MIMIC-CXR-LT every study has exactly one finding, so π is a map from the 19 finding labels to expert ids.

## The GRASP partition (`submitted` in `partitions/maps.json`)

| expert | train samples | pathology frequency ranks | labels |
|---|---:|---|---|
| E0 | 11,506 | 5–11 | Support Devices, Edema, Pneumonia, Pneumothorax, Lung Lesion, Fracture, Enlarged Cardiomediastinum |
| E1 | 9,243 | 1, 12–18 | Lung Opacity, Consolidation, Pleural Other, Calcification of the Aorta, Tortuous Aorta, Pneumoperitoneum, Subcutaneous Emphysema, Pneumomediastinum |
| E2 | 13,484 | 2–4 | Cardiomegaly, Atelectasis, Pleural Effusion |
| E3 | 53,260 | – | No Finding (reserved healthy expert) |

Construction: all *No Finding* studies are routed to a reserved healthy expert; the 18 pathology labels
are split into three groups by a balanced METIS partition (ufactor 500, i.e. ~1.5:1 size tolerance) of a
Jaccard k-nearest-neighbour graph over the label sets of the samples. Because the data are single-label,
this is a balanced binning of whole labels; it is approximately frequency-ordered (1.46:1 instead of 16:1
for contiguous frequency bins). It is not the official head/medium/tail split of MIMIC-CXR-LT.

```bash
# exact paper partition (byte-identical routing files)
python partitions/build_label_files.py --map submitted --csv $GRASP_DATA/mimic-lt/mimic-cxr-lt_train.csv \
    --out $GRASP_DATA/mimic-lt/labels_gt.txt
# rebuild with METIS from scratch (needs `pip install metis datasketch` and libmetis; the result can differ across METIS builds)
python partitions/metis_label_partition.py --csv $GRASP_DATA/mimic-lt/mimic-cxr-lt_train.csv --out labels_metis.txt
# recover the map of any routing file
python partitions/build_label_files.py --derive_from labels_gt.txt --csv mimic-cxr-lt_train.csv
```

## Sensitivity variants (Table 4a, App. D)

All variants keep whole-label assignment and are trained at GRASP-LowDim width with generator seed 0.

| map | experts | construction |
|---|---:|---|
| `random0`, `random1`, `random2` | 4 | healthy expert kept; pathology labels randomly re-assigned to three bins with the target loads of `submitted` |
| `k3` | 3 | two pathology groups + healthy |
| `k5` | 5 | four pathology groups + healthy |
| `hmt` | 3 | the official head / medium / tail frequency strata of MIMIC-CXR-LT (71-sample tail) |
| `oneperlabel` | 19 | one expert per label |

Test-split routing files use the same maps (`--trailing_newline` reproduces the paper files exactly).
The run files cover Tables 1–3 only; a variant's routing files are built with
`python partitions/build_label_files.py --map <variant> --csv <split csv> --out <file>`.

## Text-embedding partition (Embed, MoCLE, ImageNet-LT)

`partitions/text_clusters.py`: bisecting k-means (cosine, K = 4, n_init 100, max_iter 1000, seed 42) on the
pooled FLUX text embeddings of the training prompts. Writes `labels.txt` and `centroids.npz`; at generation
time test prompts are assigned to the nearest centroid by the `textlabel` / `mocle` routers.
