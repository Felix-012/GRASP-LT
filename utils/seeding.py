"""Noise seeds for generated images."""


def noise_seed(p: int, seed: int, mode: str = "per_sample", start: int = 0, batch_size: int = 1) -> int:
    """Seed of global image position `p`.

    per_sample: every image gets its own seed (seed + p); independent of batch size, world size, resume offset.
    per_batch:  blocks of `batch_size` consecutive positions (counted from the resume offset `start`) share
                one seed.
    """
    if mode == "per_batch":
        return int(seed) + (int(start) // int(batch_size)) + ((int(p) - int(start)) // int(batch_size))
    if mode != "per_sample":
        raise ValueError(f"unknown seed mode {mode!r}")
    return int(seed) + int(p)
