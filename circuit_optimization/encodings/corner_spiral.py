import numpy as np


def get_permutation(m: int, n: int) -> np.ndarray:
    """Spiral scan starting from the top-left corner moving right."""
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive integers")
    top, bottom = 0, m - 1
    left, right = 0, n - 1
    order = []
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            order.append(top * n + c)
        top += 1
        for r in range(top, bottom + 1):
            order.append(r * n + right)
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                order.append(bottom * n + c)
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                order.append(r * n + left)
            left += 1
    return np.array(order, dtype=np.int64)
