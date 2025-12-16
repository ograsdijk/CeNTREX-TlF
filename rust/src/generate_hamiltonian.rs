use crate::constants::{XConstants, BConstants};
use crate::states::{UncoupledBasisState, UncoupledState, CoupledBasisState, CoupledState};
use crate::x_uncoupled;
use crate::b_coupled;
use num_complex::Complex64;

/// Container for X state Hamiltonian matrices.
pub struct HamiltonianUncoupledX {
    pub h_ff: Vec<Complex64>,
    pub h_sx: Vec<Complex64>,
    pub h_sy: Vec<Complex64>,
    pub h_sz: Vec<Complex64>,
    pub h_zx: Vec<Complex64>,
    pub h_zy: Vec<Complex64>,
    pub h_zz: Vec<Complex64>,
}

/// Container for B state Hamiltonian matrices.
pub struct HamiltonianCoupledB {
    pub h_rot: Vec<Complex64>,
    pub h_mhf_tl: Vec<Complex64>,
    pub h_mhf_f: Vec<Complex64>,
    pub h_ld: Vec<Complex64>,
    pub h_cp1_tl: Vec<Complex64>,
    pub h_c_tl: Vec<Complex64>,
    pub h_sx: Vec<Complex64>,
    pub h_sy: Vec<Complex64>,
    pub h_sz: Vec<Complex64>,
    pub h_zx: Vec<Complex64>,
    pub h_zy: Vec<Complex64>,
    pub h_zz: Vec<Complex64>,
}

/// Calculate matrix elements for a given operator and basis.
///
/// Optimized to precompute operator application on all basis states.
pub fn h_mat_elems(
    h: fn(UncoupledBasisState, &XConstants) -> UncoupledState,
    qn: &[UncoupledBasisState],
    constants: &XConstants,
) -> Vec<Complex64> {
    let n = qn.len();
    let mut result = vec![Complex64::new(0.0, 0.0); n * n];

    // Optimization: Precompute H|b> for all basis states.
    // This reduces operator applications from O(N^2) to O(N).
    let h_applied: Vec<UncoupledState> = qn.iter()
        .map(|b| h(*b, constants))
        .collect();

    for (i, a) in qn.iter().enumerate() {
        for j in i..n {
            // Instead of recomputing h(qn[j]), access the precomputed state
            let psi = &h_applied[j];

            // Calculate <a | psi>
            let mut val = Complex64::new(0.0, 0.0);
            // Iterate by reference to avoid cloning
            for (amp, basis) in &psi.terms {
                if *basis == *a {
                    val += amp;
                }
            }

            result[i * n + j] = val;
            if i != j {
                result[j * n + i] = val.conj();
            }
        }
    }
    result
}

/// Calculate matrix elements for a given operator and basis (Coupled B state).
pub fn h_mat_elems_b(
    h: fn(CoupledBasisState, &BConstants) -> CoupledState,
    qn: &[CoupledBasisState],
    constants: &BConstants,
) -> Vec<Complex64> {
    let n = qn.len();
    let mut result = vec![Complex64::new(0.0, 0.0); n * n];

    let h_applied: Vec<CoupledState> = qn.iter()
        .map(|b| h(*b, constants))
        .collect();

    for (i, a) in qn.iter().enumerate() {
        for j in i..n {
            let psi = &h_applied[j];
            let mut val = Complex64::new(0.0, 0.0);
            for (amp, basis) in &psi.terms {
                if *basis == *a {
                    val += amp;
                }
            }
            result[i * n + j] = val;
            if i != j {
                result[j * n + i] = val.conj();
            }
        }
    }
    result
}

/// Generate all X state Hamiltonian matrices.
pub fn generate_uncoupled_hamiltonian_x(
    qn: &[UncoupledBasisState],
    constants: &XConstants,
) -> HamiltonianUncoupledX {
    HamiltonianUncoupledX {
        h_ff: h_mat_elems(x_uncoupled::h_ff, qn, constants),
        h_sx: h_mat_elems(x_uncoupled::h_sx, qn, constants),
        h_sy: h_mat_elems(x_uncoupled::h_sy, qn, constants),
        h_sz: h_mat_elems(x_uncoupled::h_sz, qn, constants),
        h_zx: h_mat_elems(x_uncoupled::h_zx, qn, constants),
        h_zy: h_mat_elems(x_uncoupled::h_zy, qn, constants),
        h_zz: h_mat_elems(x_uncoupled::h_zz, qn, constants),
    }
}

/// Generate all B state Hamiltonian matrices.
pub fn generate_coupled_hamiltonian_b(
    qn: &[CoupledBasisState],
    constants: &BConstants,
) -> HamiltonianCoupledB {
    HamiltonianCoupledB {
        h_rot: h_mat_elems_b(b_coupled::h_rot, qn, constants),
        h_mhf_tl: h_mat_elems_b(b_coupled::h_mhf_tl, qn, constants),
        h_mhf_f: h_mat_elems_b(b_coupled::h_mhf_f, qn, constants),
        h_ld: h_mat_elems_b(b_coupled::h_ld, qn, constants),
        h_cp1_tl: h_mat_elems_b(b_coupled::h_cp1_tl, qn, constants),
        h_c_tl: h_mat_elems_b(b_coupled::h_c_tl, qn, constants),
        h_sx: h_mat_elems_b(b_coupled::h_sx, qn, constants),
        h_sy: h_mat_elems_b(b_coupled::h_sy, qn, constants),
        h_sz: h_mat_elems_b(b_coupled::h_sz, qn, constants),
        h_zx: h_mat_elems_b(b_coupled::h_zx, qn, constants),
        h_zy: h_mat_elems_b(b_coupled::h_zy, qn, constants),
        h_zz: h_mat_elems_b(b_coupled::h_zz, qn, constants),
    }
}
