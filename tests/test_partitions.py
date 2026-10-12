import json
from collections import Counter, defaultdict
from pathlib import Path

MAPS = json.loads((Path(__file__).resolve().parents[1] / "partitions" / "maps.json").read_text())
TRAIN_COUNTS = {  # MIMIC-CXR-LT training split
    "No Finding": 53260, "Lung Opacity": 7927, "Cardiomegaly": 5113, "Atelectasis": 4539, "Pleural Effusion": 3832,
    "Support Devices": 3279, "Edema": 2395, "Pneumonia": 2195, "Pneumothorax": 1172, "Lung Lesion": 1036,
    "Fracture": 791, "Enlarged Cardiomediastinum": 638, "Consolidation": 609, "Pleural Other": 254,
    "Calcification of the Aorta": 207, "Tortuous Aorta": 175, "Pneumoperitoneum": 32,
    "Subcutaneous Emphysema": 27, "Pneumomediastinum": 12}


def sizes(name):
    out = defaultdict(int)
    for label, e in MAPS[name].items():
        out[e] += TRAIN_COUNTS[label]
    return dict(out)


def test_every_map_covers_all_labels():
    for name, m in MAPS.items():
        assert set(m) == set(TRAIN_COUNTS), name


def test_submitted_partition_matches_the_paper():
    assert sizes("submitted") == {0: 11506, 1: 9243, 2: 13484, 3: 53260}
    assert MAPS["submitted"]["No Finding"] == 3


def test_expert_counts():
    expected = {"submitted": 4, "random0": 4, "random1": 4, "random2": 4, "k3": 3, "k5": 5, "hmt": 3, "oneperlabel": 19}
    for name, k in expected.items():
        assert len(set(MAPS[name].values())) == k, name


def test_hmt_tail_has_71_samples():
    assert min(sizes("hmt").values()) == 71


def test_smallest_expert_sizes_reported_in_table4():
    assert min(sizes("submitted").values()) == 9243
    assert min(sizes("k3").values()) == 17080
    assert min(sizes("k5").values()) == 8431
    assert min(sizes("oneperlabel").values()) == 12
    # Table 4 reports the smallest expert over the three random maps (10,505 / 7,927 / 9,733)
    assert min(min(sizes(f"random{i}").values()) for i in range(3)) == 7927
