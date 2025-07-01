import numpy as np
import sympy as smp


def build_effective_operators(H0_diag, U0, HSz_unc, Hmw_free, blocks, atol=1e-12):
    """
    Build field-independent coefficient matrices for an arbitrary Ez,
    working entirely in the fixed field-free eigenbasis.

    Parameters
    ----------
    H0_diag : (n,) ndarray
        Field-free eigenvalues of the zero-field Hamiltonian.
    U0 : (n,n) ndarray
        Matrix whose columns are the field-free eigenvectors (uncoupled → field-free).
    HSz_unc : (n,n) ndarray
        Stark operator in the original uncoupled basis (d·Ê_z without Ez).
    Hmw_free : (n,n) ndarray
        Microwave coupling operator in the field-free eigenbasis.
    blocks : list of (start, stop) tuples
        Ranges of indices in the field-free basis to keep (e.g. [(0,4),(4,16)]).
    atol : float
        Tolerance for near-degenerate denominators.

    Returns
    -------
    Heff0 : (m,m) ndarray
        Zeroth-order static Hamiltonian in the kept subspace (diagonal of H0).
    Heff1 : (m,m) ndarray
        First-order Stark block P V P in the kept subspace (linear in Ez).
    Heff2 : (m,m) ndarray
        Second-order Stark folding P V Q (1/Δ) Q V P in the kept subspace (∝ Ez²).
    Hmw0 : (m,m) ndarray
        Bare microwave operator in the kept subspace.
    Hmw1 : (m,m) ndarray
        First-order microwave dressing coefficient (commutator term, ∝ Ez).
    """
    n = H0_diag.shape[0]
    # Transform Stark operator into the field-free basis
    V = U0.conj().T @ HSz_unc @ U0

    # Build mask & keep indices
    mask = np.zeros(n, dtype=bool)
    for s, t in blocks:
        mask[s:t] = True
    keep = np.where(mask)[0]

    # Zeroth-order static block and bare microwave
    Heff0 = np.diag(H0_diag[keep])
    Hmw0 = Hmw_free[np.ix_(keep, keep)]

    # First-order Stark: P V P
    Heff1_full = np.zeros((n, n), dtype=complex)
    Heff1_full[np.ix_(keep, keep)] = V[np.ix_(keep, keep)]
    Heff1 = Heff1_full[np.ix_(keep, keep)]

    # Second-order Stark: P V Q (1/Δ) Q V P
    Heff2_full = np.zeros((n, n), dtype=complex)
    E = H0_diag
    for start, stop in blocks:
        P_idx = np.arange(start, stop)
        Q_idx = np.setdiff1d(np.arange(n), P_idx)

        V_PQ = V[np.ix_(P_idx, Q_idx)]
        V_QP = V_PQ.conj().T

        E_P = E[P_idx][:, None]
        E_Q = E[Q_idx][None, :]
        Delta = E_P - E_Q
        Delta[np.abs(Delta) < atol] = np.inf

        Heff2_full[np.ix_(P_idx, P_idx)] += (V_PQ / Delta) @ V_QP

    Heff2 = Heff2_full[np.ix_(keep, keep)]

    # Microwave dressing generator S_lin
    S_lin = np.zeros((n, n), dtype=complex)
    for start, stop in blocks:
        P_idx = np.arange(start, stop)
        Q_idx = np.setdiff1d(np.arange(n), P_idx)

        E_P = E[P_idx][:, None]
        E_Q = E[Q_idx][None, :]
        Delta = E_P - E_Q
        Delta[np.abs(Delta) < atol] = np.inf

        for i, p in enumerate(P_idx):
            for j, q in enumerate(Q_idx):
                val = V[p, q] / Delta[i, j]
                S_lin[q, p] += val
                S_lin[p, q] -= np.conj(val)

    comm = Hmw_free @ S_lin - S_lin @ Hmw_free
    Hmw1 = comm[np.ix_(keep, keep)]

    return Heff0, Heff1, Heff2, Hmw0, Hmw1
