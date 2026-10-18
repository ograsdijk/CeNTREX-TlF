"""Exact complex-coordinate reachability for the structured CUDA dissipator."""

from __future__ import annotations

from typing import Any

import numpy as np


def reachable_upper(prepared: Any, packed: np.ndarray, model: tuple) -> np.ndarray:
    """Indices in np.triu_indices(n), closed under every Hamiltonian basis.

    Real and imaginary parts of each reachable coherence are retained together.
    The union of all initial supports is used, with no magnitude cutoff.
    """
    n = prepared.layout.n
    _, _, _, positions = model
    adjacency = [set() for _ in range(n)]
    for i, j in positions.T:
        adjacency[int(i)].add(int(j))
    outgoing = [set() for _ in range(n)]
    for target, entries in enumerate(prepared.incoming_transfers_by_target):
        for entry in entries:
            if entry["rate"] != 0:
                outgoing[entry["source"]].add(target)
    ui, uj = np.triu_indices(n)
    active = set()
    for i, j in zip(ui, uj, strict=True):
        columns = (
            [i]
            if i == j
            else [prepared.layout.upper_real_index(i, j), prepared.layout.upper_imag_index(i, j)]
        )
        if np.any(packed[:, columns] != 0):
            active.add((int(i), int(j)))
    frontier = list(active)
    while frontier:
        i, j = frontier.pop()
        successors = [(min(k, j), max(k, j)) for k in adjacency[i]]
        successors.extend((min(i, k), max(i, k)) for k in adjacency[j])
        if i == j:
            successors.extend((k, k) for k in outgoing[i])
        for pair in successors:
            if pair not in active:
                active.add(pair)
                frontier.append(pair)
    return np.array(
        [k for k, pair in enumerate(zip(ui, uj, strict=True)) if pair in active], dtype=np.int32
    )


def public_coordinates(n: int, upper: np.ndarray) -> np.ndarray:
    """Public packed coordinates represented by a set of complex upper entries."""
    ui, uj = np.triu_indices(n)
    wanted = np.zeros(len(ui), bool)
    wanted[upper] = True
    mapping = [i * n - i * (i - 1) // 2 for i in range(n)]
    for i, j in zip(*np.triu_indices(n, 1), strict=True):
        p = i * n - i * (i + 1) // 2 + j
        mapping.extend((p, p))
    return np.flatnonzero(wanted[np.asarray(mapping)]).astype(np.int32)
