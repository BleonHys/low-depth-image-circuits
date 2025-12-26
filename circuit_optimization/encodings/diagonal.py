import numpy as np


def get_permutation(m: int, n: int) -> np.ndarray:
    """Diagonal scan ordered by increasing (row + col)."""
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive integers")
    order = []
    for s in range(m + n - 1):
        row_start = max(0, s - (n - 1))
        row_end = min(m - 1, s)
        for r in range(row_start, row_end + 1):
            c = s - r
            order.append(r * n + c)
    return np.array(order, dtype=np.int64)
