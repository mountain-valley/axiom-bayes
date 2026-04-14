"""Observation preprocessing utilities."""

import numpy as np


def image_to_tokens(image: np.ndarray) -> np.ndarray:
    """
    Convert an (H, W, 3) RGB image into (N, 5) pixel tokens.

    Each token is [normalized_x, normalized_y, r, g, b] where spatial
    coordinates are normalized to [-1, +1].
    """
    h, w, c = image.shape
    ys, xs = np.mgrid[0:h, 0:w]
    xs_norm = 2.0 * xs / (w - 1) - 1.0
    ys_norm = 2.0 * ys / (h - 1) - 1.0
    colors = image.reshape(-1, c).astype(np.float64) / 255.0
    coords = np.stack([xs_norm.ravel(), ys_norm.ravel()], axis=1)
    return np.concatenate([coords, colors], axis=1)
