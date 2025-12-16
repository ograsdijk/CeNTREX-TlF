use crate::constants::BConstants;
use crate::states::{CoupledBasisState, CoupledState};
use crate::wigner::{wigner_3j_f, wigner_6j_f};
use num_complex::Complex64;
use std::f64::consts::SQRT_2;

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

        let ket = CoupledBasisState {
            f: f as i32,
            mf: mf as i32,
            f1: (f1 * 2.0).round() as i32,
            j: j_int,
            i1: (i1 * 2.0).round() as i32,
            i2: (i2 * 2.0).round() as i32,
            omega: omega as i32,
            electronic_state: psi.electronic_state,
            p: psi.p,
        };

        terms.push((Complex64::new(amp, 0.0), ket));
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
            * ((2.0 * j + 1.0)
                * (2.0 * jp + 1.0)
                * i1
                * (i1 + 1.0)
                * (2.0 * i1 + 1.0))
                .sqrt();

        if amp != 0.0 {
            let ket = CoupledBasisState {
                f: f as i32,
                mf: mf as i32,
                f1: (f1 * 2.0).round() as i32,
                j: j_int,
                i1: (i1 * 2.0).round() as i32,
                i2: (i2 * 2.0).round() as i32,
                omega: omega as i32,
                electronic_state: psi.electronic_state,
                p: psi.p,
            };
            terms.push((Complex64::new(amp, 0.0), ket));
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

        // F1 runs from |J - 1/2| to J + 1/2 in steps of 1
        let i1_val = 0.5_f64; // for Tl, I1 = 1/2
        let f1_min = (j - i1_val).abs();
        let f1_max = j + i1_val;

        let mut f1 = f1_min;
        while f1 <= f1_max + 1e-9 {
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
                let ket = CoupledBasisState {
                    f: f as i32,
                    mf: mf as i32,
                    f1: (f1 * 2.0).round() as i32,
                    j: j_int,
                    i1: (i1 * 2.0).round() as i32,
                    i2: (i2 * 2.0).round() as i32,
                    omega: omega as i32,
                    electronic_state: psi.electronic_state,
                    p: psi.p,
                };
                terms.push((Complex64::new(amp, 0.0), ket));
            }

            f1 += 1.0;
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

    let ket = CoupledBasisState {
        f: f as i32,
        mf: mf as i32,
        f1: (f1 * 2.0).round() as i32,
        j: j as i32,
        i1: (i1 * 2.0).round() as i32,
        i2: (i2 * 2.0).round() as i32,
        omega: omega as i32,
        electronic_state: psi.electronic_state,
        p: psi.p,
    };

    CoupledState::from_vec(vec![(Complex64::new(amp, 0.0), ket)])
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

/// Electric dipole operator (p-th spherical tensor component) for B state.
/// Rust translation of Python d_p.
pub fn d_p(psi: CoupledBasisState, p: i32, constants: &BConstants) -> CoupledState {
    let jp = psi.j as f64;
    let i1p = psi.i1 as f64 / 2.0;
    let i2p = psi.i2 as f64 / 2.0;
    let f1p = psi.f1 as f64 / 2.0;
    let fp = psi.f as f64;
    let mfp = psi.mf as f64;
    let omegap = psi.omega as f64;

    let i1 = i1p;
    let i2 = i2p;
    let mf = mfp + p as f64;
    let omega = omegap;

    let mut terms = Vec::new();

    let jp_int = psi.j;
    let j_min = (jp_int - 1).abs();
    let j_max = jp_int + 1;

    for j_int in j_min..=j_max {
        let j = j_int as f64;

        // F1 range: |J - I1| ... J + I1 in steps of 1
        let f1_min = (j - i1).abs();
        let f1_max = j + i1;
        let mut f1 = f1_min;
        while f1 <= f1_max + 1e-9 {
            // F range: |F1 - I2| ... F1 + I2 in steps of 1
            let f_min = (f1 - i2).abs();
            let f_max = f1 + i2;
            let mut f = f_min;
            while f <= f_max + 1e-9 {
                let exp1 =
                    (f + fp + f1 + f1p + i1 + i2 - omega - mf).round() as i32;
                let phase1 = if exp1 % 2 == 0 { 1.0 } else { -1.0 };

                let amp = constants.mu_e
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
                    let ket = CoupledBasisState {
                        f: f.round() as i32,
                        mf: mf.round() as i32,
                        f1: (f1 * 2.0).round() as i32,
                        j: j_int,
                        i1: (i1 * 2.0).round() as i32,
                        i2: (i2 * 2.0).round() as i32,
                        omega: omega as i32,
                        electronic_state: psi.electronic_state,
                        p: psi.p,
                    };
                    terms.push((Complex64::new(amp, 0.0), ket));
                }

                f += 1.0;
            }

            f1 += 1.0;
        }
    }

    CoupledState::from_vec(terms)
}

/// Stark Hamiltonian for E along x: H_Sx = -(d_-1 - d_+1)/sqrt(2).
pub fn h_sx(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    // -(d_-1 - d_+1) = d_+1 - d_-1
    let res = d_p(psi, 1, constants) - d_p(psi, -1, constants);
    res / SQRT_2
}

/// Stark Hamiltonian for E along y: H_Sy = -i(d_-1 + d_+1)/sqrt(2).
pub fn h_sy(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let sum = d_p(psi, -1, constants) + d_p(psi, 1, constants);
    let minus_i = Complex64::new(0.0, -1.0);
    sum * minus_i / SQRT_2
}

/// Stark Hamiltonian for E along z: H_Sz = -d_0.
pub fn h_sz(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    d_p(psi, 0, constants) * -1.0
}

/// Magnetic dipole operator (p-th spherical tensor component) for B state.
/// Rust translation of Python mu_p (with gL = 1.0 hard-coded).
pub fn mu_p(psi: CoupledBasisState, p: i32, constants: &BConstants) -> CoupledState {
    let g_l = 1.0;

    let jp = psi.j as f64;
    let i1p = psi.i1 as f64 / 2.0;
    let i2p = psi.i2 as f64 / 2.0;
    let f1p = psi.f1 as f64 / 2.0;
    let fp = psi.f as f64;
    let mfp = psi.mf as f64;
    let omegap = psi.omega as f64;

    let i1 = i1p;
    let i2 = i2p;
    let mf = mfp + p as f64;
    let omega = omegap;

    let mut terms = Vec::new();

    let jp_int = psi.j;
    let j_min = (jp_int - 1).abs();
    let j_max = jp_int + 1;

    for j_int in j_min..=j_max {
        let j = j_int as f64;

        let f1_min = (j - i1).abs();
        let f1_max = j + i1;
        let mut f1 = f1_min;
        while f1 <= f1_max + 1e-9 {
            let f_min = (f1 - i2).abs();
            let f_max = f1 + i2;
            let mut f = f_min;
            while f <= f_max + 1e-9 {
                let exp1 =
                    (f + fp + f1 + f1p + i1 + i2 - omega - mf).round() as i32;
                let phase1 = if exp1 % 2 == 0 { 1.0 } else { -1.0 };

                let amp = g_l
                    * omega
                    * constants.mu_b
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
                    let ket = CoupledBasisState {
                        f: f.round() as i32,
                        mf: mf.round() as i32,
                        f1: (f1 * 2.0).round() as i32,
                        j: j_int,
                        i1: (i1 * 2.0).round() as i32,
                        i2: (i2 * 2.0).round() as i32,
                        omega: omega as i32,
                        electronic_state: psi.electronic_state,
                        p: psi.p,
                    };
                    terms.push((Complex64::new(amp, 0.0), ket));
                }

                f += 1.0;
            }

            f1 += 1.0;
        }
    }

    CoupledState::from_vec(terms)
}

/// Zeeman Hamiltonian for B along x: H_Zx = -(μ_-1 - μ_+1)/sqrt(2).
pub fn h_zx(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let res = mu_p(psi, 1, constants) - mu_p(psi, -1, constants);
    res / SQRT_2
}

/// Zeeman Hamiltonian for B along y: H_Zy = -i(μ_-1 + μ_+1)/sqrt(2).
pub fn h_zy(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    let sum = mu_p(psi, -1, constants) + mu_p(psi, 1, constants);
    let minus_i = Complex64::new(0.0, -1.0);
    sum * minus_i / SQRT_2
}

/// Zeeman Hamiltonian for B along z: H_Zz = -μ_0.
pub fn h_zz(psi: CoupledBasisState, constants: &BConstants) -> CoupledState {
    mu_p(psi, 0, constants) * -1.0
}

