import numpy as np


def get_permutation(m: int, n: int) -> np.ndarray:
    """Row-wise boustrophedon ordering (snake)."""
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive integers")
    grid = np.arange(m * n, dtype=np.int64).reshape(m, n)
    grid[1::2] = grid[1::2, ::-1]
    return grid.reshape(-1)
