import numpy as np


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def get_permutation(m: int, n: int) -> np.ndarray:
    """Morton (Z-order) scan. Requires square, power-of-two images."""
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive integers")
    if m != n or not _is_power_of_two(m):
        raise ValueError(
            "morton ordering requires square power-of-two images (m == n and power of 2)"
        )
    bits = int(np.log2(m))
    indices = np.arange(m * n, dtype=np.int64)
    rows = indices // n
    cols = indices % n

    codes = np.zeros_like(indices)
    for b in range(bits):
        codes |= ((cols >> b) & 1) << (2 * b)
        codes |= ((rows >> b) & 1) << (2 * b + 1)

    return indices[np.argsort(codes, kind="stable")]
