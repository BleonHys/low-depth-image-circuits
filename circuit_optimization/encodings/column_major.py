import numpy as np


def get_permutation(m: int, n: int) -> np.ndarray:
    """Column-major ordering."""
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive integers")
    return np.arange(m * n, dtype=np.int64).reshape(m, n).T.reshape(-1)
