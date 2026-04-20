use crate::b_coupled;
use crate::constants::{BConstants, XConstants};
use crate::states::{CoupledBasisState, CoupledState, UncoupledBasisState, UncoupledState};
use crate::x_uncoupled;
use num_complex::Complex64;
use rayon::prelude::*;

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
    let h_applied: Vec<UncoupledState> = qn.iter().map(|b| h(*b, constants)).collect();

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

    let h_applied: Vec<CoupledState> = qn.iter().map(|b| h(*b, constants)).collect();

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

pub fn generate_uncoupled_hamiltonian_x(
    qn: &[UncoupledBasisState],
    constants: &XConstants,
) -> HamiltonianUncoupledX {
    let ops: Vec<fn(UncoupledBasisState, &XConstants) -> UncoupledState> = vec![
        x_uncoupled::h_ff,
        x_uncoupled::h_sx,
        x_uncoupled::h_sy,
        x_uncoupled::h_sz,
        x_uncoupled::h_zx,
        x_uncoupled::h_zy,
        x_uncoupled::h_zz,
    ];

    let results: Vec<Vec<Complex64>> = ops
        .into_par_iter()
        .map(|op| h_mat_elems(op, qn, constants))
        .collect();

    let mut it = results.into_iter();
    HamiltonianUncoupledX {
        h_ff: it.next().unwrap(),
        h_sx: it.next().unwrap(),
        h_sy: it.next().unwrap(),
        h_sz: it.next().unwrap(),
        h_zx: it.next().unwrap(),
        h_zy: it.next().unwrap(),
        h_zz: it.next().unwrap(),
    }
}

pub fn generate_coupled_hamiltonian_b(
    qn: &[CoupledBasisState],
    constants: &BConstants,
) -> HamiltonianCoupledB {
    let ops: Vec<fn(CoupledBasisState, &BConstants) -> CoupledState> = vec![
        b_coupled::h_rot,
        b_coupled::h_mhf_tl,
        b_coupled::h_mhf_f,
        b_coupled::h_ld,
        b_coupled::h_cp1_tl,
        b_coupled::h_c_tl,
        b_coupled::h_sx,
        b_coupled::h_sy,
        b_coupled::h_sz,
        b_coupled::h_zx,
        b_coupled::h_zy,
        b_coupled::h_zz,
    ];

    let results: Vec<Vec<Complex64>> = ops
        .into_par_iter()
        .map(|op| h_mat_elems_b(op, qn, constants))
        .collect();

    let mut it = results.into_iter();
    HamiltonianCoupledB {
        h_rot: it.next().unwrap(),
        h_mhf_tl: it.next().unwrap(),
        h_mhf_f: it.next().unwrap(),
        h_ld: it.next().unwrap(),
        h_cp1_tl: it.next().unwrap(),
        h_c_tl: it.next().unwrap(),
        h_sx: it.next().unwrap(),
        h_sy: it.next().unwrap(),
        h_sz: it.next().unwrap(),
        h_zx: it.next().unwrap(),
        h_zy: it.next().unwrap(),
        h_zz: it.next().unwrap(),
    }
}
