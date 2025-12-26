import numpy as np


def get_permutation(m: int, n: int) -> np.ndarray:
    """JPEG-like zigzag diagonal scan."""
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive integers")
    order = []
    for s in range(m + n - 1):
        row_start = max(0, s - (n - 1))
        row_end = min(m - 1, s)
        rows = range(row_start, row_end + 1)
        if s % 2 == 0:
            rows = reversed(list(rows))
        for r in rows:
            c = s - r
            order.append(r * n + c)
    return np.array(order, dtype=np.int64)
