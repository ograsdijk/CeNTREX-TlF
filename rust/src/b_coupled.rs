use crate::constants::BConstants;
use crate::states::{CoupledBasisState, CoupledState};
use crate::wigner::{wigner_3j_f, wigner_6j_f};
use num_complex::Complex64;
use std::f64::consts::SQRT_2;

#[inline]
fn make_ket(
    psi: &CoupledBasisState,
    j: i32,
    f: f64,
    mf: f64,
    f1: f64,
    i1: f64,
    i2: f64,
    omega: f64,
) -> CoupledBasisState {
    CoupledBasisState {
        f: f.round() as i32,
        mf: mf.round() as i32,
        f1: (f1 * 2.0).round() as i32,
        j,
        i1: (i1 * 2.0).round() as i32,
        i2: (i2 * 2.0).round() as i32,
        omega: omega as i32,
        electronic_state: psi.electronic_state,
        parity: psi.parity,
    }
}

fn spherical_tensor_matrix_element(
    psi: &CoupledBasisState,
    p: i32,
    prefactor: f64,
) -> CoupledState {
    let jp = psi.j as f64;
    let i1 = psi.i1 as f64 / 2.0;
    let i2 = psi.i2 as f64 / 2.0;
    let f1p = psi.f1 as f64 / 2.0;
    let fp = psi.f as f64;
    let mfp = psi.mf as f64;
    let omegap = psi.omega as f64;
    let omega = omegap;
    let mf = mfp + p as f64;

    let mut terms = Vec::new();

    let jp_int = psi.j;
    let j_min = (jp_int - 1).abs();
    let j_max = jp_int + 1;

    for j_int in j_min..=j_max {
        let j = j_int as f64;

        let f1_min = (j - i1).abs();
        let f1_max = j + i1;
        let f1_steps = (f1_max - f1_min).round() as i32;

        for f1_step in 0..=f1_steps {
            let f1 = f1_min + f1_step as f64;
            let f_min = (f1 - i2).abs();
            let f_max = f1 + i2;
            let f_steps = (f_max - f_min).round() as i32;

            for f_step in 0..=f_steps {
                let f = f_min + f_step as f64;
                let exp1 = (f + fp + f1 + f1p + i1 + i2 - omega - mf).round() as i32;
                let phase1 = if exp1 % 2 == 0 { 1.0 } else { -1.0 };

                let amp = prefactor
                    * phase1
                    * ((2.0 * f + 1.0)
                        * (2.0 * fp + 1.0)
                        * (2.0 * f1 + 1.0)
                        * (2.0 * f1p + 1.0)
                        * (2.0 * j + 1.0)
                        * (2.0 * jp + 1.0))
                        .sqrt()
                    * wigner_3j_f(f, 1.0, fp, -mf, p as f64, mfp)
                    * wigner_3j_f(j, 1.0, jp, -omega, 0.0, omegap)
                    * wigner_6j_f(f1p, fp, i2, f, f1, 1.0)
                    * wigner_6j_f(jp, f1p, i1, f1, j, 1.0);

                if amp != 0.0 {
                    terms.push((
                        Complex64::new(amp, 0.0),
                        make_ket(psi, j_int, f, mf, f1, i1, i2, omega),
                    ));
                }
            }
        }
    }

    CoupledState::from_vec(terms)
}

/// Lambda-doubling q-term operator for B state in coupled Omega basis.
/// Rust translation of Python H_LD in your B-state module.
pub fn h_ld(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let j = psi.j as f64;
    let omega_new = -psi.omega;

    // amp = q * J * (J + 1) / 2
    let amp = constants.q * j * (j + 1.0) / 2.0;

    let mut ket = psi;
    ket.omega = omega_new;

    CoupledState::from_vec(vec![(Complex64::new(amp, 0.0), ket)])
}

/// Lambda-doubling nuclear spin-rotation operator for Tl nucleus in B state.
/// Rust translation of Python H_cp1_Tl.
pub fn h_cp1_tl(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    // Physical quantum numbers (half-integers where appropriate)
    let jp = psi.j as f64;
    let i1p = psi.i1 as f64 / 2.0;
    let i2p = psi.i2 as f64 / 2.0;
    let f1p = psi.f1 as f64 / 2.0;
    let fp = psi.f as f64;
    let mfp = psi.mf as f64;
    let omegap = psi.omega as f64;

    // I1, I2, F, F1, mF the same in bra and ket
    let i1 = i1p;
    let i2 = i2p;
    let f1 = f1p;
    let f = fp;
    let mf = mfp;

    // Omegas are opposite
    let omega = -omegap;
    let q = omega;

    let mut terms = Vec::new();

    // J runs from |Jp - 1| to Jp + 1 in integer steps
    let jp_int = psi.j;
    let j_min = (jp_int - 1).abs();
    let j_max = jp_int + 1;

    for j_int in j_min..=j_max {
        let j = j_int as f64;

        // phase = (-1)^(J + Jp + F1 + I1 - Omegap)
        let exp1 = (j + jp + f1 + i1 - omegap).round() as i32;
        let phase1 = if exp1 % 2 == 0 { 1.0 } else { -1.0 };

        let sj = wigner_6j_f(i1, jp, f1, j, i1, 1.0);
        let prefac = -constants.c1p_tl / 2.0
            * phase1
            * sj
            * (i1 * (i1 + 1.0) * (2.0 * i1 + 1.0) * (2.0 * j + 1.0) * (2.0 * jp + 1.0)).sqrt();

        // First term in brackets
        let exp_j = j as i32;
        let phase_j = if exp_j % 2 == 0 { 1.0 } else { -1.0 };

        let t1 = phase_j
            * (j * (j + 1.0) * (2.0 * j + 1.0)).sqrt()
            * wigner_3j_f(j, 1.0, jp, 0.0, q, omegap)
            * wigner_3j_f(j, 1.0, j, -omega, q, 0.0);

        // Second term in brackets
        let exp_jp = jp as i32;
        let phase_jp = if exp_jp % 2 == 0 { 1.0 } else { -1.0 };

        let t2 = phase_jp
            * (jp * (jp + 1.0) * (2.0 * jp + 1.0)).sqrt()
            * wigner_3j_f(jp, 1.0, jp, 0.0, q, omegap)
            * wigner_3j_f(j, 1.0, jp, -omega, q, 0.0);

        let amp = prefac * (t1 + t2);

        terms.push((
            Complex64::new(amp, 0.0),
            make_ket(&psi, j_int, f, mf, f1, i1, i2, omega),
        ));
    }

    CoupledState::from_vec(terms)
}

/// Magnetic hyperfine operator for Tl nucleus in B state.
/// Rust translation of Python H_mhf_Tl.
pub fn h_mhf_tl(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let jp = psi.j as f64;
    let i1p = psi.i1 as f64 / 2.0;
    let i2p = psi.i2 as f64 / 2.0;
    let f1p = psi.f1 as f64 / 2.0;
    let fp = psi.f as f64;
    let mfp = psi.mf as f64;
    let omegap = psi.omega as f64;

    let i1 = i1p;
    let i2 = i2p;
    let f1 = f1p;
    let f = fp;
    let mf = mfp;
    let omega = omegap;

    let mut terms = Vec::new();

    let jp_int = psi.j;
    let j_min = (jp_int - 1).abs();
    let j_max = jp_int + 1;

    for j_int in j_min..=j_max {
        let j = j_int as f64;

        let exp1 = (j + jp + f1 + i1 - omega).round() as i32;
        let phase1 = if exp1 % 2 == 0 { 1.0 } else { -1.0 };

        let amp = omega
            * constants.h1_tl
            * phase1
            * wigner_6j_f(i1, jp, f1, j, i1, 1.0)
            * wigner_3j_f(j, 1.0, jp, -omega, 0.0, omegap)
            * ((2.0 * j + 1.0) * (2.0 * jp + 1.0) * i1 * (i1 + 1.0) * (2.0 * i1 + 1.0)).sqrt();

        if amp != 0.0 {
            terms.push((
                Complex64::new(amp, 0.0),
                make_ket(&psi, j_int, f, mf, f1, i1, i2, omega),
            ));
        }
    }

    CoupledState::from_vec(terms)
}

/// Magnetic hyperfine operator for F nucleus in B state.
/// Rust translation of Python H_mhf_F.
pub fn h_mhf_f(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let jp = psi.j as f64;
    let i1p = psi.i1 as f64 / 2.0;
    let i2p = psi.i2 as f64 / 2.0;
    let f1p = psi.f1 as f64 / 2.0;
    let fp = psi.f as f64;
    let mfp = psi.mf as f64;
    let omegap = psi.omega as f64;

    let i1 = i1p;
    let i2 = i2p;
    let f = fp;
    let mf = mfp;
    let omega = omegap;

    let mut terms = Vec::new();

    let jp_int = psi.j;
    let j_min = (jp_int - 1).abs();
    let j_max = jp_int + 1;

    for j_int in j_min..=j_max {
        let j = j_int as f64;

        let f1_min = (j - i1).abs();
        let f1_max = j + i1;
        let f1_steps = (f1_max - f1_min).round() as i32;

        for f1_step in 0..=f1_steps {
            let f1 = f1_min + f1_step as f64;
            let exp1 = (2.0 * f1p + f + 2.0 * j + i1 + i2 - omega + 1.0).round() as i32;
            let phase1 = if exp1 % 2 == 0 { 1.0 } else { -1.0 };

            let amp = omega
                * constants.h1_f
                * phase1
                * wigner_6j_f(i2, f1p, f, f1, i2, 1.0)
                * wigner_6j_f(jp, f1p, i1, f1, j, 1.0)
                * wigner_3j_f(j, 1.0, jp, -omega, 0.0, omegap)
                * ((2.0 * f1 + 1.0)
                    * (2.0 * f1p + 1.0)
                    * (2.0 * j + 1.0)
                    * (2.0 * jp + 1.0)
                    * i2
                    * (i2 + 1.0)
                    * (2.0 * i2 + 1.0))
                    .sqrt();

            if amp != 0.0 {
                terms.push((
                    Complex64::new(amp, 0.0),
                    make_ket(&psi, j_int, f, mf, f1, i1, i2, omega),
                ));
            }
        }
    }

    CoupledState::from_vec(terms)
}

/// Nuclear spin-rotation operator for Tl nucleus in B state.
/// Rust translation of Python H_c_Tl.
pub fn h_c_tl(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let jp = psi.j as f64;
    let i1p = psi.i1 as f64 / 2.0;
    let i2p = psi.i2 as f64 / 2.0;
    let f1p = psi.f1 as f64 / 2.0;
    let fp = psi.f as f64;
    let mfp = psi.mf as f64;
    let omegap = psi.omega as f64;

    let j = jp;
    let i1 = i1p;
    let i2 = i2p;
    let f1 = f1p;
    let f = fp;
    let mf = mfp;
    let omega = omegap;

    let exp1 = (i1 + f1 + j).round() as i32;
    let phase1 = if exp1 % 2 == 0 { 1.0 } else { -1.0 };

    let amp = constants.c_tl
        * phase1
        * wigner_6j_f(jp, i1, f1, i1, j, 1.0)
        * (i1 * (i1 + 1.0) * (2.0 * i1 + 1.0) * j * (j + 1.0) * (2.0 * j + 1.0)).sqrt();

    CoupledState::from_vec(vec![(
        Complex64::new(amp, 0.0),
        make_ket(&psi, j as i32, f, mf, f1, i1, i2, omega),
    )])
}

/// Rotational Hamiltonian: B*J^2 + D*J^4 + H*J^6
pub fn h_rot(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let j = psi.j as f64;
    let j2 = j * (j + 1.0);
    let j4 = j2 * j2;
    let j6 = j4 * j2;

    // Matches your current Python: B*J2 + D*J4 + H*J6
    let amp = constants.b_rot * j2 - constants.d_rot * j4 + constants.h_const * j6;

    CoupledState::from_vec(vec![(Complex64::new(amp, 0.0), psi)])
}

fn d_p(psi: CoupledBasisState, p: i32, constants: &BConstants) -> CoupledState {
    spherical_tensor_matrix_element(&psi, p, constants.mu_e)
}

/// Stark components (x, y, z) built from one d_p evaluation per spherical component.
pub fn stark_components(
    psi: CoupledBasisState,
    constants: &BConstants,
) -> (CoupledState, CoupledState, CoupledState) {
    let d_minus = d_p(psi, -1, constants);
    let d_zero = d_p(psi, 0, constants);
    let d_plus = d_p(psi, 1, constants);

    let h_sx = (d_plus.clone() - d_minus.clone()) / SQRT_2;
    let h_sy = (d_minus + d_plus) * -Complex64::I / SQRT_2;
    let h_sz = d_zero * -1.0;
    (h_sx, h_sy, h_sz)
}

/// Stark Hamiltonian for E along x: H_Sx = (d_{+1} - d_{-1}) / sqrt(2).
#[allow(dead_code)]
pub fn h_sx(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    stark_components(psi, constants).0
}

/// Stark Hamiltonian for E along y: H_Sy = -i(d_{-1} + d_{+1}) / sqrt(2).
#[allow(dead_code)]
pub fn h_sy(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    stark_components(psi, constants).1
}

/// Stark Hamiltonian for E along z: H_Sz = -d_0.
#[allow(dead_code)]
pub fn h_sz(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    stark_components(psi, constants).2
}

fn mu_p(psi: CoupledBasisState, p: i32, constants: &BConstants) -> CoupledState {
    let prefactor = constants.gl * psi.omega as f64 * constants.mu_b;
    spherical_tensor_matrix_element(&psi, p, prefactor)
}

/// Zeeman components (x, y, z) built from one mu_p evaluation per spherical component.
pub fn zeeman_components(
    psi: CoupledBasisState,
    constants: &BConstants,
) -> (CoupledState, CoupledState, CoupledState) {
    let mu_minus = mu_p(psi, -1, constants);
    let mu_zero = mu_p(psi, 0, constants);
    let mu_plus = mu_p(psi, 1, constants);

    let h_zx = (mu_plus.clone() - mu_minus.clone()) / SQRT_2;
    let h_zy = (mu_minus + mu_plus) * -Complex64::I / SQRT_2;
    let h_zz = mu_zero * -1.0;
    (h_zx, h_zy, h_zz)
}

/// Zeeman Hamiltonian for B along x: H_Zx = (mu_{+1} - mu_{-1}) / sqrt(2).
#[allow(dead_code)]
pub fn h_zx(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    zeeman_components(psi, constants).0
}

/// Zeeman Hamiltonian for B along y: H_Zy = -i(mu_{-1} + mu_{+1}) / sqrt(2).
#[allow(dead_code)]
pub fn h_zy(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    zeeman_components(psi, constants).1
}

/// Zeeman Hamiltonian for B along z: H_Zz = -mu_0.
#[allow(dead_code)]
pub fn h_zz(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    zeeman_components(psi, constants).2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::states::ElectronicState;

    fn make_b_state(j: i32, f: i32, mf: i32, f1: i32, omega: i32) -> CoupledBasisState {
        CoupledBasisState {
            j,
            f,
            mf,
            i1: 1, // 2*I1 = 1 (Tl, I=1/2)
            i2: 1, // 2*I2 = 1 (F, I=1/2)
            f1,    // 2*F1
            omega,
            parity: None,
            electronic_state: ElectronicState::B,
        }
    }

    fn default_constants() -> BConstants {
        BConstants::default()
    }

    #[test]
    fn test_h_rot_j1() {
        let psi = make_b_state(1, 1, 0, 1, 1);
        let result = h_rot(psi, &default_constants());
        // B*J(J+1) - D*J(J+1)^2 + H*J(J+1)^3 for J=1
        let c = default_constants();
        let j2 = 2.0;
        let expected = c.b_rot * j2 - c.d_rot * j2 * j2 + c.h_const * j2 * j2 * j2;
        assert_eq!(result.terms.len(), 1);
        assert!((result.terms[0].0.re - expected).abs() < 1e-6);
        assert_eq!(result.terms[0].1, psi);
    }

    #[test]
    fn test_h_rot_preserves_quantum_numbers() {
        let psi = make_b_state(2, 2, 1, 3, 1);
        let result = h_rot(psi, &default_constants());
        assert_eq!(result.terms.len(), 1);
        assert_eq!(result.terms[0].1, psi);
    }

    #[test]
    fn test_h_ld_flips_omega() {
        let psi = make_b_state(1, 1, 0, 1, 1);
        let result = h_ld(psi, &default_constants());
        assert_eq!(result.terms.len(), 1);
        assert_eq!(result.terms[0].1.omega, -1);
    }

    #[test]
    fn test_h_ld_amplitude() {
        let psi = make_b_state(1, 1, 0, 1, 1);
        let c = default_constants();
        let result = h_ld(psi, &c);
        let expected = c.q * 1.0 * 2.0 / 2.0; // q * J * (J+1) / 2 for J=1
        assert!((result.terms[0].0.re - expected).abs() < 1e-6);
    }

    #[test]
    fn test_h_mhf_tl_hermiticity() {
        let states: Vec<CoupledBasisState> = vec![
            make_b_state(1, 1, 0, 1, 1),
            make_b_state(1, 1, 0, 3, 1),
            make_b_state(1, 2, 0, 3, 1),
        ];
        let c = default_constants();
        let n = states.len();
        let mut matrix = vec![Complex64::ZERO; n * n];
        for (j, ket) in states.iter().enumerate() {
            let result = h_mhf_tl(*ket, &c);
            for (amp, basis) in &result.terms {
                for (i, bra) in states.iter().enumerate() {
                    if *bra == *basis {
                        matrix[i * n + j] += amp;
                    }
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                let diff = (matrix[i * n + j] - matrix[j * n + i].conj()).norm();
                assert!(
                    diff < 1e-10,
                    "H_mhf_Tl not Hermitian at ({i},{j}): diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_h_c_tl_diagonal() {
        let psi = make_b_state(1, 1, 0, 1, 1);
        let c = default_constants();
        let result = h_c_tl(psi, &c);
        assert_eq!(result.terms.len(), 1);
        assert_eq!(result.terms[0].1, psi);
    }

    #[test]
    fn test_stark_components_exist() {
        let psi = make_b_state(1, 1, 0, 1, 1);
        let c = default_constants();
        let _sx = h_sx(psi, &c);
        let _sy = h_sy(psi, &c);
        let _sz = h_sz(psi, &c);
    }

    #[test]
    fn test_zeeman_components_exist() {
        let psi = make_b_state(1, 1, 0, 1, 1);
        let c = default_constants();
        let _zx = h_zx(psi, &c);
        let _zy = h_zy(psi, &c);
        let _zz = h_zz(psi, &c);
    }
}
