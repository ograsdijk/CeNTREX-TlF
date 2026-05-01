"""RF Ramsey benchmark simulation (TlF X state, fixed-basis pure-state propagation).

Experimental, isolated from the main centrex_tlf code path. See the demo notebook
and validation script in the parent directory for usage.
"""

from .fields import AnalyticDCField, AnalyticRFRegion, FieldStack, MagneticRFRegion
from .hamiltonian import build_basis, build_H_func
from .observables import per_j_populations, survival_probability
from .propagator import (
    PropagationResult,
    SegmentedGridReport,
    build_segmented_t_grid,
    propagate_midpoint,
    step_eigh,
)
from .propagator_truncated import (
    propagate_midpoint_truncated,
    select_subspace_J_manifold,
    select_subspace_by_overlap,
    step_eigh_truncated,
)
from .scan import ScanResult, ScanSpec, run_scan
from .simulator import RamseyRFConfig, RamseyRFResult, RamseyRFSimulator
from .states import (
    adiabatic_dressed_initial_states,
    dressed_initial_states,
    j_manifold_indices,
    select_uncoupled,
    targets_to_state_vectors,
)
from .trajectory import BallisticTrajectory

__all__ = [
    "AnalyticDCField",
    "AnalyticRFRegion",
    "MagneticRFRegion",
    "FieldStack",
    "BallisticTrajectory",
    "build_basis",
    "build_H_func",
    "dressed_initial_states",
    "adiabatic_dressed_initial_states",
    "j_manifold_indices",
    "select_uncoupled",
    "targets_to_state_vectors",
    "survival_probability",
    "per_j_populations",
    "PropagationResult",
    "propagate_midpoint",
    "step_eigh",
    "SegmentedGridReport",
    "build_segmented_t_grid",
    "propagate_midpoint_truncated",
    "step_eigh_truncated",
    "select_subspace_by_overlap",
    "select_subspace_J_manifold",
    "RamseyRFConfig",
    "RamseyRFResult",
    "RamseyRFSimulator",
    "ScanSpec",
    "ScanResult",
    "run_scan",
]
