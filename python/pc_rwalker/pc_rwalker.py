import numpy as np

from . import _pc_rwalker


def random_walker_segmentation(
    xyz: np.ndarray,
    seed_indices: list[list[int]],
    n_neighbors: int,
    *,
    n_neighbors_graph: int = 15,
    sigma1: float = 1.0,
    sigma2: float = 1.0,
    min_weight: float = 0.0001,
    n_proc: int = -1,
    return_flat: bool = True
) -> list[list[int]] | np.ndarray: 
    indices = _pc_rwalker.random_walker_segmentation(
        xyz,
        seed_indices,
        n_neighbors,
        n_neighbors_graph,
        sigma1,
        sigma2,
        min_weight,
        n_proc,
    )

    if not return_flat:
        return indices
    
    flat_idx = np.zeros(xyz.shape[0], dtype=np.int32)
    for label, idx in enumerate(indices):
        flat_idx[idx] = label

    return flat_idx
