import numpy as np
import matplotlib.pyplot as plt

from circuit_optimization.encodings.registry import ENCODINGS


def draw_permutation_grid(m: int, n: int, permutation_func, permutation_name, label_pixels: bool = True, arrow_cuts: float = 0.15):
    perm = permutation_func(m, n)

    # Map linear index -> (row, col)
    coords = np.array([(idx // n, idx % n) for idx in range(m * n)])

    fig, ax = plt.subplots(figsize=(n, m))

    # Draw grid
    for i in range(m + 1):
        ax.plot([0, n], [i, i], linewidth=1, color='black')

    for j in range(n + 1):
        ax.plot([j, j], [0, m], linewidth=1, color='black')

    # Draw arrows following permutation
    for a, b in zip(perm[:-1], perm[1:]):
        r1, c1 = coords[a]
        r2, c2 = coords[b]

        start_x = c1 + 0.5
        start_y = m - r1 - 0.5

        end_x = c2 + 0.5
        end_y = m - r2 - 0.5

        dx = end_x - start_x
        dy = end_y - start_y

        if dx != 0:
            start_x += arrow_cuts * np.sign(dx)
            end_x -= arrow_cuts * np.sign(dx)

        if dy != 0:
            start_y += arrow_cuts * np.sign(dy)
            end_y -= arrow_cuts * np.sign(dy)

        dx = end_x - start_x
        dy = end_y - start_y

        ax.arrow(
            start_x, start_y,
            dx, dy,
            head_width=0.15,
            length_includes_head=True,
            fc='red',
            ec='red'
        )

    # Label pixels
    if label_pixels:
        for idx, (r, c) in enumerate(coords):
            ax.text(c + 0.5, m - r - 0.5, str(idx),
                    ha='center', va='center')

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, n)
    ax.set_ylim(0, m)
    ax.set_title(permutation_name)

    plt.show()


RENAMED_ENCODINGS = {
    "Row Major": ENCODINGS.get("row_major"),
    "Column Major": ENCODINGS.get("column_major"),
    "Horizontal Snake": ENCODINGS.get("snake"),
    "Vertical Snake": ENCODINGS.get("vertical_snake"),
    "Left Diagonal": ENCODINGS.get("diagonal"),
    "Zig-Zag Diagonal": ENCODINGS.get("diagonal_zigzag"),
    "Corner Spiral": ENCODINGS.get("corner_spiral"),
    "Morton": ENCODINGS.get("morton"),
    "Hilbert": ENCODINGS.get("hilbert"),
}


for permutation_name, permutation_func in RENAMED_ENCODINGS.items():
    draw_permutation_grid(m=4, n=4, permutation_func=permutation_func, permutation_name=permutation_name)