from utils.seeding import noise_seed


def test_per_sample_is_unique():
    seeds = [noise_seed(p, 5868, "per_sample") for p in range(1000)]
    assert len(set(seeds)) == 1000 and seeds[0] == 5868


def test_per_batch_blocks_of_eight_share_a_seed():
    seeds = [noise_seed(p, 0, "per_batch", start=0, batch_size=8) for p in range(24)]
    assert seeds == [0] * 8 + [1] * 8 + [2] * 8


def test_per_batch_matches_legacy_formula_with_resume_offset():
    seed, start, bs = 9090, 13, 8
    for p in range(start, start + 100):
        assert noise_seed(p, seed, "per_batch", start=start, batch_size=bs) == seed + start // bs + (p - start) // bs
