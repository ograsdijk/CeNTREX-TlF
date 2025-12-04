use num_complex::Complex64;
use std::f64::consts::SQRT_2;
use crate::states::{CoupledState, CoupledBasisState};
use crate::wigner::{wigner_3j_f, wigner_6j_f};

/// Convert Cartesian polarization vector to spherical tensor components.
/// Returns [epsilon_-1, epsilon_0, epsilon_+1]
fn polarization_to_spherical(pol: &[Complex64; 3]) -> [Complex64; 3] {
    let ex = pol[0];
    let ey = pol[1];
    let ez = pol[2];

    let e0 = ez;
    let e_plus = -(ex + Complex64::new(0.0, 1.0) * ey) / SQRT_2;
    let e_minus = (ex - Complex64::new(0.0, 1.0) * ey) / SQRT_2;

    // Return in order corresponding to p = -1, 0, 1
    [e_minus, e0, e_plus]
}

/// Calculate electric dipole matrix element between two basis states for a specific spherical component p.
/// Computes < bra | d_p | ket >
fn calculate_ed_me_basis_component(
    bra: &CoupledBasisState,
    ket: &CoupledBasisState,
    p: i32,
) -> Complex64 {
    // Quantum numbers for bra
    let J_bra = bra.J as f64;
    let I1_bra = bra.I1 as f64 / 2.0;
    let I2_bra = bra.I2 as f64 / 2.0;
    let F1_bra = bra.F1 as f64 / 2.0;
    let F_bra = bra.F as f64;
    let mF_bra = bra.mF as f64;
    let Omega_bra = bra.Omega as f64;

    // Quantum numbers for ket
    let J_ket = ket.J as f64;
    let I1_ket = ket.I1 as f64 / 2.0;
    let I2_ket = ket.I2 as f64 / 2.0;
    let F1_ket = ket.F1 as f64 / 2.0;
    let F_ket = ket.F as f64;
    let mF_ket = ket.mF as f64;
    let Omega_ket = ket.Omega as f64;

    // Selection rules check: mF_bra = mF_ket + p
    if (mF_bra - (mF_ket + p as f64)).abs() > 1e-9 {
        return Complex64::new(0.0, 0.0);
    }

    // Nuclear spins must match
    if (I1_bra - I1_ket).abs() > 1e-9 || (I2_bra - I2_ket).abs() > 1e-9 {
        return Complex64::new(0.0, 0.0);
    }

    let I1 = I1_ket;
    let I2 = I2_ket;

    // q in molecule frame: Omega_bra - Omega_ket
    let q_mol = Omega_bra - Omega_ket;

    // Phase factor (-1)^(F + F' + F1 + F1' + I1 + I2 - Omega - mF)
    let exp_val = F_bra + F_ket + F1_bra + F1_ket + I1 + I2 - Omega_bra - mF_bra;
    let phase = if (exp_val.round() as i32) % 2 == 0 { 1.0 } else { -1.0 };

    let prefactor = ((2.0 * F_bra + 1.0)
        * (2.0 * F_ket + 1.0)
        * (2.0 * F1_bra + 1.0)
        * (2.0 * F1_ket + 1.0)
        * (2.0 * J_bra + 1.0)
        * (2.0 * J_ket + 1.0))
        .sqrt();

    let val = phase
        * prefactor
        * wigner_3j_f(F_bra, 1.0, F_ket, -mF_bra, p as f64, mF_ket)
        * wigner_3j_f(J_bra, 1.0, J_ket, -Omega_bra, q_mol, Omega_ket)
        * wigner_6j_f(F1_ket, F_ket, I2, F_bra, F1_bra, 1.0)
        * wigner_6j_f(J_ket, F1_ket, I1, F1_bra, J_bra, 1.0);

    // This is the full (non-reduced) matrix element for given p
    Complex64::new(val, 0.0)
}

fn calculate_ed_me_basis_reduced(
    bra: &CoupledBasisState,
    ket: &CoupledBasisState,
) -> Complex64 {
    // Quantum numbers for bra
    let J_bra = bra.J as f64;
    let I1_bra = bra.I1 as f64 / 2.0;
    let I2_bra = bra.I2 as f64 / 2.0;
    let F1_bra = bra.F1 as f64 / 2.0;
    let F_bra = bra.F as f64;
    let Omega_bra = bra.Omega as f64;

    // Quantum numbers for ket
    let J_ket = ket.J as f64;
    let I1_ket = ket.I1 as f64 / 2.0;
    let I2_ket = ket.I2 as f64 / 2.0;
    let F1_ket = ket.F1 as f64 / 2.0;
    let F_ket = ket.F as f64;
    let Omega_ket = ket.Omega as f64;

    // Nuclear spins must match (operator does not act on nuclear spins)
    if (I1_bra - I1_ket).abs() > 1e-9 || (I2_bra - I2_ket).abs() > 1e-9 {
        return Complex64::new(0.0, 0.0);
    }

    let I1 = I1_ket;
    let I2 = I2_ket;

    // q in molecule frame: Ω - Ω'
    let q_mol = Omega_bra - Omega_ket;
    if q_mol.abs() >= 2.0 {
        // selection rule |ΔΩ| < 2
        return Complex64::new(0.0, 0.0);
    }

    // Phase factor (-1)^(F1' + F1 + F' + I1 + I2 - Ω)
    let exp_val = F1_ket + F1_bra + F_ket + I1 + I2 - Omega_bra;
    let phase = if (exp_val.round() as i32) % 2 == 0 { 1.0 } else { -1.0 };

    let prefactor = ((2.0 * J_bra + 1.0)
        * (2.0 * J_ket + 1.0)
        * (2.0 * F1_bra + 1.0)
        * (2.0 * F1_ket + 1.0)
        * (2.0 * F_bra + 1.0)
        * (2.0 * F_ket + 1.0))
        .sqrt();

    let val = phase
        * prefactor
        * wigner_6j_f(F1_ket, F_ket, I2, F_bra, F1_bra, 1.0)
        * wigner_6j_f(J_ket, F1_ket, I1, F1_bra, J_bra, 1.0)
        * wigner_3j_f(J_bra, 1.0, J_ket, -Omega_bra, q_mol, Omega_ket);

    Complex64::new(val, 0.0)
}

/// Calculate < bra | d . epsilon | ket > for mixed states (superpositions).
pub fn generate_ed_me_mixed_state(
    bra: &CoupledState,
    ket: &CoupledState,
    pol_vec: &[Complex64; 3],
    reduced_dipole: f64,
    reduced: bool,
) -> Complex64 {
    if reduced {
        // Reduced ME: ignore polarization and m_F structure
        let mut total_me = Complex64::new(0.0, 0.0);
        for (c_bra, state_bra) in bra.iter() {
            for (c_ket, state_ket) in ket.iter() {
                let me = calculate_ed_me_basis_reduced(state_bra, state_ket);
                total_me += c_bra.conj() * c_ket * me;
            }
        }
        total_me
    } else {
        // Full matrix element including angular + polarization dependence
        let eps_sph = polarization_to_spherical(pol_vec);
        let mut total_me = Complex64::new(0.0, 0.0);

        for (c_bra, state_bra) in bra.iter() {
            for (c_ket, state_ket) in ket.iter() {
                // p = -1
                let me_m1 = calculate_ed_me_basis_component(state_bra, state_ket, -1);
                total_me += c_bra.conj() * c_ket * (-1.0) * eps_sph[2] * me_m1;

                // p = 0
                let me_0 = calculate_ed_me_basis_component(state_bra, state_ket, 0);
                total_me += c_bra.conj() * c_ket * eps_sph[1] * me_0;

                // p = +1
                let me_p1 = calculate_ed_me_basis_component(state_bra, state_ket, 1);
                total_me += c_bra.conj() * c_ket * (-1.0) * eps_sph[0] * me_p1;
            }
        }
        total_me
    }
}

/// Generate the optical coupling matrix for transitions between quantum states.
/// Returns a flattened n x n matrix (row-major).
pub fn generate_coupling_matrix(
    qn: &[CoupledState],
    ground_states: &[CoupledState],
    excited_states: &[CoupledState],
    pol_vec: &[Complex64; 3],
    reduced_dipole: f64,
    reduced: bool,
) -> Vec<Complex64> {
    let n = qn.len();
    let mut h = vec![Complex64::new(0.0, 0.0); n * n];

    // Assume `pol_vec` is already normalized on the Python side.

    for gs in ground_states {
        // Find index of ground state in QN basis
        if let Some(i) = qn.iter().position(|s| s == gs) {
            for es in excited_states {
                // Find index of excited state in QN basis
                if let Some(j) = qn.iter().position(|s| s == es) {
                    // Calculate matrix element < es | d·ε | gs >
                    let elem = generate_ed_me_mixed_state(es, gs, pol_vec, reduced_dipole, reduced);

                    // H[i, j] = elem * reduced_dipole
                    h[i * n + j] = elem * reduced_dipole;

                    // Hermitian conjugate H[j, i] = elem*
                    h[j * n + i] = elem.conj() * reduced_dipole;
                }
            }
        }
    }

    h
}
