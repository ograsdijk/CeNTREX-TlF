from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def solution_to_density_matrices(solution: solve_ivp, n_states: int) -> np.ndarray:
    return solution.y.T.reshape((-1, n_states, n_states))


def scattering_signal(
    rho_t: np.ndarray,
    jump_rate_operator: np.ndarray,
) -> np.ndarray:
    return np.real(np.einsum("tij,ji->t", rho_t, jump_rate_operator))


def integrated_scattering_probability(
    t_eval: np.ndarray,
    rho_t: np.ndarray,
    jump_rate_operator: np.ndarray,
) -> float:
    return float(np.trapezoid(scattering_signal(rho_t, jump_rate_operator), x=t_eval))
