use crate::constants::BConstants;
use crate::states::{CoupledBasisState, CoupledState};
use crate::wigner::{wigner_3j_f, wigner_6j_f};
use num_complex::Complex64;
use std::f64::consts::SQRT_2;

/// Lambda-doubling q-term operator for B state in coupled Omega basis.
/// Rust translation of Python H_LD in your B-state module.
pub fn H_LD(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let J = psi.J as f64;
    let Omega_new = -psi.Omega;

    // amp = q * J * (J + 1) / 2
    let amp = constants.q * J * (J + 1.0) / 2.0;

    let mut ket = psi;
    ket.Omega = Omega_new;

    CoupledState::from_vec(vec![(Complex64::new(amp, 0.0), ket)])
}

/// Lambda-doubling nuclear spin-rotation operator for Tl nucleus in B state.
/// Rust translation of Python H_cp1_Tl.
pub fn H_cp1_Tl(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    // Physical quantum numbers (half-integers where appropriate)
    let Jp = psi.J as f64;
    let I1p = psi.I1 as f64 / 2.0;
    let I2p = psi.I2 as f64 / 2.0;
    let F1p = psi.F1 as f64 / 2.0;
    let Fp = psi.F as f64;
    let mFp = psi.mF as f64;
    let Omegap = psi.Omega as f64;

    // I1, I2, F, F1, mF the same in bra and ket
    let I1 = I1p;
    let I2 = I2p;
    let F1 = F1p;
    let F = Fp;
    let mF = mFp;

    // Omegas are opposite
    let Omega = -Omegap;
    let q = Omega;

    let mut terms = Vec::new();

    // J runs from |Jp - 1| to Jp + 1 in integer steps
    let Jp_int = psi.J;
    let j_min = (Jp_int - 1).abs();
    let j_max = Jp_int + 1;

    for J_int in j_min..=j_max {
        let J = J_int as f64;

        // phase = (-1)^(J + Jp + F1 + I1 - Omegap)
        let exp1 = (J + Jp + F1 + I1 - Omegap).round() as i32;
        let phase1 = if exp1 % 2 == 0 { 1.0 } else { -1.0 };

        let sj = wigner_6j_f(I1, Jp, F1, J, I1, 1.0);
        let prefac = -constants.c1p_Tl / 2.0
            * phase1
            * sj
            * (I1 * (I1 + 1.0) * (2.0 * I1 + 1.0) * (2.0 * J + 1.0) * (2.0 * Jp + 1.0)).sqrt();

        // First term in brackets
        let exp_j = J as i32;
        let phase_j = if exp_j % 2 == 0 { 1.0 } else { -1.0 };

        let t1 = phase_j
            * (J * (J + 1.0) * (2.0 * J + 1.0)).sqrt()
            * wigner_3j_f(J, 1.0, Jp, 0.0, q, Omegap)
            * wigner_3j_f(J, 1.0, J, -Omega, q, 0.0);

        // Second term in brackets
        let exp_jp = Jp as i32;
        let phase_jp = if exp_jp % 2 == 0 { 1.0 } else { -1.0 };

        let t2 = phase_jp
            * (Jp * (Jp + 1.0) * (2.0 * Jp + 1.0)).sqrt()
            * wigner_3j_f(Jp, 1.0, Jp, 0.0, q, Omegap)
            * wigner_3j_f(J, 1.0, Jp, -Omega, q, 0.0);

        let amp = prefac * (t1 + t2);

        let ket = CoupledBasisState {
            F: F as i32,
            mF: mF as i32,
            F1: (F1 * 2.0).round() as i32,
            J: J_int,
            I1: (I1 * 2.0).round() as i32,
            I2: (I2 * 2.0).round() as i32,
            Omega: Omega as i32,
            electronic_state: psi.electronic_state,
            P: psi.P,
        };

        terms.push((Complex64::new(amp, 0.0), ket));
    }

    CoupledState::from_vec(terms)
}

/// Magnetic hyperfine operator for Tl nucleus in B state.
/// Rust translation of Python H_mhf_Tl.
pub fn H_mhf_Tl(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let Jp = psi.J as f64;
    let I1p = psi.I1 as f64 / 2.0;
    let I2p = psi.I2 as f64 / 2.0;
    let F1p = psi.F1 as f64 / 2.0;
    let Fp = psi.F as f64;
    let mFp = psi.mF as f64;
    let Omegap = psi.Omega as f64;

    let I1 = I1p;
    let I2 = I2p;
    let F1 = F1p;
    let F = Fp;
    let mF = mFp;
    let Omega = Omegap;

    let mut terms = Vec::new();

    let Jp_int = psi.J;
    let j_min = (Jp_int - 1).abs();
    let j_max = Jp_int + 1;

    for J_int in j_min..=j_max {
        let J = J_int as f64;

        let exp1 = (J + Jp + F1 + I1 - Omega).round() as i32;
        let phase1 = if exp1 % 2 == 0 { 1.0 } else { -1.0 };

        let amp = Omega
            * constants.h1_Tl
            * phase1
            * wigner_6j_f(I1, Jp, F1, J, I1, 1.0)
            * wigner_3j_f(J, 1.0, Jp, -Omega, 0.0, Omegap)
            * ((2.0 * J + 1.0)
                * (2.0 * Jp + 1.0)
                * I1
                * (I1 + 1.0)
                * (2.0 * I1 + 1.0))
                .sqrt();

        if amp != 0.0 {
            let ket = CoupledBasisState {
                F: F as i32,
                mF: mF as i32,
                F1: (F1 * 2.0).round() as i32,
                J: J_int,
                I1: (I1 * 2.0).round() as i32,
                I2: (I2 * 2.0).round() as i32,
                Omega: Omega as i32,
                electronic_state: psi.electronic_state,
                P: psi.P,
            };
            terms.push((Complex64::new(amp, 0.0), ket));
        }
    }

    CoupledState::from_vec(terms)
}

/// Magnetic hyperfine operator for F nucleus in B state.
/// Rust translation of Python H_mhf_F.
pub fn H_mhf_F(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let Jp = psi.J as f64;
    let I1p = psi.I1 as f64 / 2.0;
    let I2p = psi.I2 as f64 / 2.0;
    let F1p = psi.F1 as f64 / 2.0;
    let Fp = psi.F as f64;
    let mFp = psi.mF as f64;
    let Omegap = psi.Omega as f64;

    let I1 = I1p;
    let I2 = I2p;
    let F = Fp;
    let mF = mFp;
    let Omega = Omegap;

    let mut terms = Vec::new();

    let Jp_int = psi.J;
    let j_min = (Jp_int - 1).abs();
    let j_max = Jp_int + 1;

    for J_int in j_min..=j_max {
        let J = J_int as f64;

        // F1 runs from |J - 1/2| to J + 1/2 in steps of 1
        let I1_val = 0.5_f64; // for Tl, I1 = 1/2
        let f1_min = (J - I1_val).abs();
        let f1_max = J + I1_val;

        let mut F1 = f1_min;
        while F1 <= f1_max + 1e-9 {
            let exp1 = (2.0 * F1p + F + 2.0 * J + I1 + I2 - Omega + 1.0).round() as i32;
            let phase1 = if exp1 % 2 == 0 { 1.0 } else { -1.0 };

            let amp = Omega
                * constants.h1_F
                * phase1
                * wigner_6j_f(I2, F1p, F, F1, I2, 1.0)
                * wigner_6j_f(Jp, F1p, I1, F1, J, 1.0)
                * wigner_3j_f(J, 1.0, Jp, -Omega, 0.0, Omegap)
                * ((2.0 * F1 + 1.0)
                    * (2.0 * F1p + 1.0)
                    * (2.0 * J + 1.0)
                    * (2.0 * Jp + 1.0)
                    * I2
                    * (I2 + 1.0)
                    * (2.0 * I2 + 1.0))
                    .sqrt();

            if amp != 0.0 {
                let ket = CoupledBasisState {
                    F: F as i32,
                    mF: mF as i32,
                    F1: (F1 * 2.0).round() as i32,
                    J: J_int,
                    I1: (I1 * 2.0).round() as i32,
                    I2: (I2 * 2.0).round() as i32,
                    Omega: Omega as i32,
                    electronic_state: psi.electronic_state,
                    P: psi.P,
                };
                terms.push((Complex64::new(amp, 0.0), ket));
            }

            F1 += 1.0;
        }
    }

    CoupledState::from_vec(terms)
}

/// Nuclear spin-rotation operator for Tl nucleus in B state.
/// Rust translation of Python H_c_Tl.
pub fn H_c_Tl(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let Jp = psi.J as f64;
    let I1p = psi.I1 as f64 / 2.0;
    let I2p = psi.I2 as f64 / 2.0;
    let F1p = psi.F1 as f64 / 2.0;
    let Fp = psi.F as f64;
    let mFp = psi.mF as f64;
    let Omegap = psi.Omega as f64;

    let J = Jp;
    let I1 = I1p;
    let I2 = I2p;
    let F1 = F1p;
    let F = Fp;
    let mF = mFp;
    let Omega = Omegap;

    let exp1 = (I1 + F1 + J).round() as i32;
    let phase1 = if exp1 % 2 == 0 { 1.0 } else { -1.0 };

    let amp = constants.c_Tl
        * phase1
        * wigner_6j_f(Jp, I1, F1, I1, J, 1.0)
        * (I1 * (I1 + 1.0) * (2.0 * I1 + 1.0) * J * (J + 1.0) * (2.0 * J + 1.0)).sqrt();

    let ket = CoupledBasisState {
        F: F as i32,
        mF: mF as i32,
        F1: (F1 * 2.0).round() as i32,
        J: J as i32,
        I1: (I1 * 2.0).round() as i32,
        I2: (I2 * 2.0).round() as i32,
        Omega: Omega as i32,
        electronic_state: psi.electronic_state,
        P: psi.P,
    };

    CoupledState::from_vec(vec![(Complex64::new(amp, 0.0), ket)])
}

/// Rotational Hamiltonian: B*J^2 + D*J^4 + H*J^6
pub fn Hrot(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let J = psi.J as f64;
    let j2 = J * (J + 1.0);
    let j4 = j2 * j2;
    let j6 = j4 * j2;

    // Matches your current Python: B*J2 + D*J4 + H*J6
    let amp = constants.B_rot * j2 - constants.D_rot * j4 + constants.H_const * j6;

    CoupledState::from_vec(vec![(Complex64::new(amp, 0.0), psi)])
}

/// Electric dipole operator (p-th spherical tensor component) for B state.
/// Rust translation of Python d_p.
pub fn d_p(psi: CoupledBasisState, p: i32, constants: &BConstants) -> CoupledState {
    let Jp = psi.J as f64;
    let I1p = psi.I1 as f64 / 2.0;
    let I2p = psi.I2 as f64 / 2.0;
    let F1p = psi.F1 as f64 / 2.0;
    let Fp = psi.F as f64;
    let mFp = psi.mF as f64;
    let Omegap = psi.Omega as f64;

    let I1 = I1p;
    let I2 = I2p;
    let mF = mFp + p as f64;
    let Omega = Omegap;

    let mut terms = Vec::new();

    let Jp_int = psi.J;
    let j_min = (Jp_int - 1).abs();
    let j_max = Jp_int + 1;

    for J_int in j_min..=j_max {
        let J = J_int as f64;

        // F1 range: |J - I1| ... J + I1 in steps of 1
        let f1_min = (J - I1).abs();
        let f1_max = J + I1;
        let mut F1 = f1_min;
        while F1 <= f1_max + 1e-9 {
            // F range: |F1 - I2| ... F1 + I2 in steps of 1
            let f_min = (F1 - I2).abs();
            let f_max = F1 + I2;
            let mut F = f_min;
            while F <= f_max + 1e-9 {
                let exp1 =
                    (F + Fp + F1 + F1p + I1 + I2 - Omega - mF).round() as i32;
                let phase1 = if exp1 % 2 == 0 { 1.0 } else { -1.0 };

                let amp = constants.mu_E
                    * phase1
                    * ((2.0 * F + 1.0)
                        * (2.0 * Fp + 1.0)
                        * (2.0 * F1 + 1.0)
                        * (2.0 * F1p + 1.0)
                        * (2.0 * J + 1.0)
                        * (2.0 * Jp + 1.0))
                        .sqrt()
                    * wigner_3j_f(F, 1.0, Fp, -mF, p as f64, mFp)
                    * wigner_3j_f(J, 1.0, Jp, -Omega, 0.0, Omegap)
                    * wigner_6j_f(F1p, Fp, I2, F, F1, 1.0)
                    * wigner_6j_f(Jp, F1p, I1, F1, J, 1.0);

                if amp != 0.0 {
                    let ket = CoupledBasisState {
                        F: F.round() as i32,
                        mF: mF.round() as i32,
                        F1: (F1 * 2.0).round() as i32,
                        J: J_int,
                        I1: (I1 * 2.0).round() as i32,
                        I2: (I2 * 2.0).round() as i32,
                        Omega: Omega as i32,
                        electronic_state: psi.electronic_state,
                        P: psi.P,
                    };
                    terms.push((Complex64::new(amp, 0.0), ket));
                }

                F += 1.0;
            }

            F1 += 1.0;
        }
    }

    CoupledState::from_vec(terms)
}

/// Stark Hamiltonian for E along x: H_Sx = -(d_-1 - d_+1)/sqrt(2).
pub fn HSx(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    // -(d_-1 - d_+1) = d_+1 - d_-1
    let res = d_p(psi, 1, constants) - d_p(psi, -1, constants);
    res / SQRT_2
}

/// Stark Hamiltonian for E along y: H_Sy = -i(d_-1 + d_+1)/sqrt(2).
pub fn HSy(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let sum = d_p(psi, -1, constants) + d_p(psi, 1, constants);
    let minus_i = Complex64::new(0.0, -1.0);
    sum * minus_i / SQRT_2
}

/// Stark Hamiltonian for E along z: H_Sz = -d_0.
pub fn HSz(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    d_p(psi, 0, constants) * -1.0
}

/// Magnetic dipole operator (p-th spherical tensor component) for B state.
/// Rust translation of Python mu_p (with gL = 1.0 hard-coded).
pub fn mu_p(psi: CoupledBasisState, p: i32, constants: &BConstants) -> CoupledState {
    let gL = 1.0;

    let Jp = psi.J as f64;
    let I1p = psi.I1 as f64 / 2.0;
    let I2p = psi.I2 as f64 / 2.0;
    let F1p = psi.F1 as f64 / 2.0;
    let Fp = psi.F as f64;
    let mFp = psi.mF as f64;
    let Omegap = psi.Omega as f64;

    let I1 = I1p;
    let I2 = I2p;
    let mF = mFp + p as f64;
    let Omega = Omegap;

    let mut terms = Vec::new();

    let Jp_int = psi.J;
    let j_min = (Jp_int - 1).abs();
    let j_max = Jp_int + 1;

    for J_int in j_min..=j_max {
        let J = J_int as f64;

        let f1_min = (J - I1).abs();
        let f1_max = J + I1;
        let mut F1 = f1_min;
        while F1 <= f1_max + 1e-9 {
            let f_min = (F1 - I2).abs();
            let f_max = F1 + I2;
            let mut F = f_min;
            while F <= f_max + 1e-9 {
                let exp1 =
                    (F + Fp + F1 + F1p + I1 + I2 - Omega - mF).round() as i32;
                let phase1 = if exp1 % 2 == 0 { 1.0 } else { -1.0 };

                let amp = gL
                    * Omega
                    * constants.mu_B
                    * phase1
                    * ((2.0 * F + 1.0)
                        * (2.0 * Fp + 1.0)
                        * (2.0 * F1 + 1.0)
                        * (2.0 * F1p + 1.0)
                        * (2.0 * J + 1.0)
                        * (2.0 * Jp + 1.0))
                        .sqrt()
                    * wigner_3j_f(F, 1.0, Fp, -mF, p as f64, mFp)
                    * wigner_3j_f(J, 1.0, Jp, -Omega, 0.0, Omegap)
                    * wigner_6j_f(F1p, Fp, I2, F, F1, 1.0)
                    * wigner_6j_f(Jp, F1p, I1, F1, J, 1.0);

                if amp != 0.0 {
                    let ket = CoupledBasisState {
                        F: F.round() as i32,
                        mF: mF.round() as i32,
                        F1: (F1 * 2.0).round() as i32,
                        J: J_int,
                        I1: (I1 * 2.0).round() as i32,
                        I2: (I2 * 2.0).round() as i32,
                        Omega: Omega as i32,
                        electronic_state: psi.electronic_state,
                        P: psi.P,
                    };
                    terms.push((Complex64::new(amp, 0.0), ket));
                }

                F += 1.0;
            }

            F1 += 1.0;
        }
    }

    CoupledState::from_vec(terms)
}

/// Zeeman Hamiltonian for B along x: H_Zx = -(μ_-1 - μ_+1)/sqrt(2).
pub fn HZx(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let res = mu_p(psi, 1, constants) - mu_p(psi, -1, constants);
    res / SQRT_2
}

/// Zeeman Hamiltonian for B along y: H_Zy = -i(μ_-1 + μ_+1)/sqrt(2).
pub fn HZy(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let sum = mu_p(psi, -1, constants) + mu_p(psi, 1, constants);
    let minus_i = Complex64::new(0.0, -1.0);
    sum * minus_i / SQRT_2
}

/// Zeeman Hamiltonian for B along z: H_Zz = -μ_0.
pub fn HZz(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    mu_p(psi, 0, constants) * -1.0
}

