import argparse
import os
import os.path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from beyondfid.default_config import config
from beyondfid.run import run
from beyondfid.utils import update_config

mp.set_sharing_strategy('file_system')


def configure_compute_node_cache() -> None:
    """Keep model caches off $HOME, which is not reachable on Helma compute nodes."""
    workspace = os.environ.get("WORKSPACE", "").rstrip("/")
    if workspace:
        cache_root = os.path.join(workspace, ".cache")
    else:
        cache_root = os.path.join(os.getcwd(), ".cache")

    os.environ.setdefault("XDG_CACHE_HOME", cache_root)
    os.environ.setdefault("TORCH_HOME", os.path.join(cache_root, "torch"))
    os.environ.setdefault("HF_HOME", os.path.join(cache_root, "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(os.environ["HF_HOME"], "transformers"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets"))

    for key in (
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
    ):
        os.makedirs(os.environ[key], exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path', type=str, default=None)
    parser.add_argument('--results_filename', type=str, default=None)
    parser.add_argument('--data_root', type=str, default=None,
                        help='folder containing the dataset folders (default: $GRASP_DATA or $WORKSPACE/data)')
    parser.add_argument('--cxr_model', type=str, default=None,
                        help='CXR DenseNet-121 feature extractor checkpoint (default: $GRASP_CXR_MODEL or $WORKSPACE/models/m-05122024-131940.pth.tar)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='feature/result cache folder (default: <data_root>/feature_results/<dataset>)')
    parser.add_argument('--metric_seed', type=int, default=0,
                        help='seed for the random equal-size subsampling inside PRDC (precision/recall/density/'
                             'coverage); the paper runs left it unseeded, see docs/REPRODUCE.md')
    parser.add_argument('--num_synth', type=int, default=50000,
                        help='number of generated images in --gen_path (the paper uses 50,000)')
    return parser.parse_args()


def resolve_existing(label: str, candidates: list[str]) -> str:
    expanded = [os.path.expandvars(p) for p in candidates]
    for p in expanded:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not resolve {label}. Tried: {expanded}")


def extend_series(s: pd.Series, n: int) -> pd.Series:
    """Repeat s until length n, then cut off. If keep_index=False, reset to RangeIndex."""
    k = len(s)
    if k == 0:
        raise ValueError("Can't extend an empty Series.")
    if n <= 0:
        return s.iloc[0:0].copy()

    idx = np.arange(n) % k
    out = s.iloc[idx]
    return out.reset_index(drop=True).to_numpy()

if __name__ == '__main__':

    configure_compute_node_cache()
    args = parse_args()

    # decide what features to extract, run beyondfid -h for a full list, can be custom (see below)
    feature_extractors="dinov2"

     # decide what metrics you want to compute, run beyonfid -h for a full list
    metrics ="fid,prdc,irs"

    # update the config
    config = update_config(config,metrics=metrics, feature_extractors=feature_extractors)

    workspace = os.environ.get("WORKSPACE", "").rstrip("/")
    data_root = (args.data_root or os.environ.get("GRASP_DATA") or f"{workspace}/data").rstrip("/")
    if "cxr" in feature_extractors.split(","):   # ImageNet-LT uses DINOv2 features only
        config.feature_extractors.cxr.config.path = resolve_existing(
            "cxr feature model",
            [
                args.cxr_model or os.environ.get("GRASP_CXR_MODEL", ""),
                f"{workspace}/models/m-05122024-131940.pth.tar",
            ],
        )
    config.feature_extractors.compute_mode = "single"

    # manually set hyperparameters if necessary - full list is in beyondfid.default_config.py
    config.metrics.prdc.nearest_k = 10
    train_csv = resolve_existing(
        "imagenet train csv",
        [
            f"{data_root}/imagenet-lt/imagenet_lt_train.csv",
        ],
    )
    test_csv = resolve_existing(
        "imagenet test csv",
        [
            f"{data_root}/imagenet-lt/imagenet_lt_test.csv",
        ],
    )
    gen_path=args.gen_path
    output_path = args.output_path or f"{data_root}/feature_results/imagenet"
    os.makedirs(output_path, exist_ok=True)
    train_path = resolve_existing(
        "imagenet train h5",
        [
            f"{data_root}/imagenet-lt/imagenet-lt_train.h5",
        ],
    )
    test_path = resolve_existing(
        "imagenet test h5",
        [
            f"{data_root}/imagenet-lt/imagenet-lt_test.h5",
        ],
    )
    train_labels = pd.read_csv(train_csv)["Finding Labels"].to_numpy()
    test_labels = pd.read_csv(test_csv)["Finding Labels"]
    # path/to/synth can also nvi    be a generator function (see below)
    torch.manual_seed(args.metric_seed)   # PRDC subsamples the larger set with torch.randperm
    results = run(pathtrain=train_path,
                  pathtest=test_path,
                  pathsynth=gen_path,
                  output_path=output_path,
                  results_filename=args.results_filename,
                  config=config,
                  train_labels=train_labels,
                  test_labels=test_labels,
                  synth_labels=extend_series(test_labels, args.num_synth))
    # results will also be saved to "out/path/results.json", all features as tensors to out/path/<model>/*.pt
