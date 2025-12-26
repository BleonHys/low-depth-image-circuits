import numpy as np

from .row_major import get_permutation as row_major
from .column_major import get_permutation as column_major
from .snake import get_permutation as snake
from .vertical_snake import get_permutation as vertical_snake
from .diagonal import get_permutation as diagonal
from .diagonal_zigzag import get_permutation as diagonal_zigzag
from .corner_spiral import get_permutation as corner_spiral
from .morton import get_permutation as morton
from .hilbert import get_permutation as hilbert

ENCODINGS = {
    "row_major": row_major,
    "column_major": column_major,
    "snake": snake,
    "vertical_snake": vertical_snake,
    "diagonal": diagonal,
    "diagonal_zigzag": diagonal_zigzag,
    "corner_spiral": corner_spiral,
    "morton": morton,
    "hilbert": hilbert,
}

ALIASES = {
    "hierarchical": "morton",
    "zorder": "morton",
    "z-order": "morton",
    "morton": "morton",
}


def list_encodings() -> list[str]:
    """Return supported canonical encoding names."""
    return sorted(ENCODINGS.keys())


def get_permutation(m: int, n: int, name: str | None) -> np.ndarray:
    """Return a permutation for the given encoding name.

    Args:
        m: image height
        n: image width
        name: encoding name; None defaults to row_major
    """
    if name is None:
        key = "row_major"
    else:
        key = name

    key = ALIASES.get(key, key)

    if key not in ENCODINGS:
        raise ValueError(
            f"Unknown encoding '{name}'. Available encodings: {', '.join(list_encodings())}"
        )

    return ENCODINGS[key](m, n)


def inverse_permutation(idx: np.ndarray) -> np.ndarray:
    """Return the inverse permutation for idx."""
    inv = np.empty_like(idx)
    inv[idx] = np.arange(len(idx), dtype=idx.dtype)
    return inv
