import random
from typing import List, Tuple

def split_train_test_masks(indices: List[int],
                           train_pct: float,
                           test_pct: float,
                           seed: int | None = None
                          ) -> Tuple[List[int], List[int]]:
    """
    Create train/test masks from a list of indices.

    Args:
        indices: list of dataset indices.
        train_pct: fraction or percent for training (0.8 or 80).
        test_pct: fraction or percent for testing (0.2 or 20).
        seed: optional random seed.

    Returns:
        (train_mask, test_mask) — lists of 0/1 aligned with `indices`.
    """
    if seed is not None:
        random.seed(seed)

    n = len(indices)
    if n == 0:
        return [], []

    total = train_pct + test_pct
    if abs(total - 1.0) > 1e-8 and abs(total - 100.0) > 1e-8:
        raise ValueError("train_pct + test_pct must sum to 1.0 or 100.0")

    # Normalize to fractions
    if abs(total - 100.0) < 1e-8:
        train_frac = train_pct / 100.0
        test_frac = test_pct / 100.0
    else:
        train_frac = train_pct
        test_frac = test_pct

    positions = list(range(n))
    random.shuffle(positions)

    n_train = int(round(train_frac * n))
    n_test = n - n_train  # absorb rounding

    train_pos = set(positions[:n_train])
    test_pos = set(positions[n_train:])

    train_mask = [1 if i in train_pos else 0 for i in range(n)]
    test_mask  = [1 if i in test_pos else 0 for i in range(n)]

    return train_mask, test_mask