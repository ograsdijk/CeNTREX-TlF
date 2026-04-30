from __future__ import annotations

import numpy as np


def default_effective_density_matrix(model) -> np.ndarray:
    density = np.zeros((model.n_effective_states, model.n_effective_states), dtype=np.complex128)
    density[model.ground_main_index_p, model.ground_main_index_p] = 1.0
    return density
