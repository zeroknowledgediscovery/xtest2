"""Submitted geometry d_G: Euclidean distance between Fisher-score
embeddings induced by the same frozen Chow-Liu tree model used for
generation.
"""
import numpy as np


def submitted_geometry_matrix(model, discrete: np.ndarray) -> np.ndarray:
    S = model.fisher_scores(discrete)
    sq_norm = np.sum(S ** 2, axis=1)
    G = S @ S.T
    d2 = sq_norm[:, None] + sq_norm[None, :] - 2 * G
    d2 = np.clip(d2, 0, None)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    d = 0.5 * (d + d.T)
    return d
