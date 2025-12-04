use crate::constants::{XConstants, BConstants};
use crate::states::{UncoupledBasisState, UncoupledState, CoupledBasisState, CoupledState};
use crate::X_uncoupled;
use crate::B_coupled;
use num_complex::Complex64;

/// Container for X state Hamiltonian matrices.
pub struct HamiltonianUncoupledX {
    pub Hff: Vec<Complex64>,
    pub HSx: Vec<Complex64>,
    pub HSy: Vec<Complex64>,
    pub HSz: Vec<Complex64>,
    pub HZx: Vec<Complex64>,
    pub HZy: Vec<Complex64>,
    pub HZz: Vec<Complex64>,
}

/// Container for B state Hamiltonian matrices.
pub struct HamiltonianCoupledB {
    pub Hrot: Vec<Complex64>,
    pub H_mhf_Tl: Vec<Complex64>,
    pub H_mhf_F: Vec<Complex64>,
    pub H_LD: Vec<Complex64>,
    pub H_cp1_Tl: Vec<Complex64>,
    pub H_c_Tl: Vec<Complex64>,
    pub HSx: Vec<Complex64>,
    pub HSy: Vec<Complex64>,
    pub HSz: Vec<Complex64>,
    pub HZx: Vec<Complex64>,
    pub HZy: Vec<Complex64>,
    pub HZz: Vec<Complex64>,
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
pub fn h_mat_elems_B(
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
pub fn generate_uncoupled_hamiltonian_X(
    qn: &[UncoupledBasisState],
    constants: &XConstants,
) -> HamiltonianUncoupledX {
    HamiltonianUncoupledX {
        Hff: h_mat_elems(X_uncoupled::Hff, qn, constants),
        HSx: h_mat_elems(X_uncoupled::HSx, qn, constants),
        HSy: h_mat_elems(X_uncoupled::HSy, qn, constants),
        HSz: h_mat_elems(X_uncoupled::HSz, qn, constants),
        HZx: h_mat_elems(X_uncoupled::HZx, qn, constants),
        HZy: h_mat_elems(X_uncoupled::HZy, qn, constants),
        HZz: h_mat_elems(X_uncoupled::HZz, qn, constants),
    }
}

/// Generate all B state Hamiltonian matrices.
pub fn generate_coupled_hamiltonian_B(
    qn: &[CoupledBasisState],
    constants: &BConstants,
) -> HamiltonianCoupledB {
    HamiltonianCoupledB {
        Hrot: h_mat_elems_B(B_coupled::Hrot, qn, constants),
        H_mhf_Tl: h_mat_elems_B(B_coupled::H_mhf_Tl, qn, constants),
        H_mhf_F: h_mat_elems_B(B_coupled::H_mhf_F, qn, constants),
        H_LD: h_mat_elems_B(B_coupled::H_LD, qn, constants),
        H_cp1_Tl: h_mat_elems_B(B_coupled::H_cp1_Tl, qn, constants),
        H_c_Tl: h_mat_elems_B(B_coupled::H_c_Tl, qn, constants),
        HSx: h_mat_elems_B(B_coupled::HSx, qn, constants),
        HSy: h_mat_elems_B(B_coupled::HSy, qn, constants),
        HSz: h_mat_elems_B(B_coupled::HSz, qn, constants),
        HZx: h_mat_elems_B(B_coupled::HZx, qn, constants),
        HZy: h_mat_elems_B(B_coupled::HZy, qn, constants),
        HZz: h_mat_elems_B(B_coupled::HZz, qn, constants),
    }
}
