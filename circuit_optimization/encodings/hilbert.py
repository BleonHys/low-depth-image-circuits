import numpy as np


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _d2xy(n: int, d: int) -> tuple[int, int]:
    """Convert Hilbert distance d to (x, y) for an n x n grid."""
    x = 0
    y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def get_permutation(m: int, n: int) -> np.ndarray:
    """Hilbert space-filling curve scan. Requires square, power-of-two images."""
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive integers")
    if m != n or not _is_power_of_two(m):
        raise ValueError(
            "hilbert ordering requires square power-of-two images (m == n and power of 2)"
        )
    order = np.empty(m * n, dtype=np.int64)
    for d in range(m * n):
        x, y = _d2xy(m, d)
        order[d] = y * n + x
    return order
