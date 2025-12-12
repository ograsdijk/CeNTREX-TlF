use crate::constants::XConstants;
use crate::states::{UncoupledBasisState, UncoupledState};
use crate::quantum_operators::*;
use num_complex::Complex64;

/// Calculate parity of X state.
pub fn parity_x(j: i32) -> i8 {
    if j % 2 == 0 { 1 } else { -1 }
}

/// Hyperfine term c1: c1 * I1 . J
pub fn h_c1(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    constants.c1 * (
        com(i1z, jz, psi, constants)
        + 0.5 * (com(i1p, jm, psi, constants) + com(i1m, jp, psi, constants))
    )
}

/// Hyperfine term c2: c2 * I2 . J
pub fn h_c2(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    constants.c2 * (
        com(i2z, jz, psi, constants)
        + 0.5 * (com(i2p, jm, psi, constants) + com(i2m, jp, psi, constants))
    )
}

/// Hyperfine term c4: c4 * I1 . I2
pub fn h_c4(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    constants.c4 * (
        com(i1z, i2z, psi, constants)
        + 0.5 * (com(i1p, i2m, psi, constants) + com(i1m, i2p, psi, constants))
    )
}

/// Hyperfine term c3a.
pub fn h_c3a(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    let factor = 15.0 * constants.c3 / constants.c1 / constants.c2 / ((2 * psi.j + 3) * (2 * psi.j - 1)) as f64;
    factor * com(h_c1, h_c2, psi, constants)
}

/// Hyperfine term c3b.
pub fn h_c3b(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    let factor = 15.0 * constants.c3 / constants.c2 / constants.c1 / ((2 * psi.j + 3) * (2 * psi.j - 1)) as f64;
    factor * com(h_c2, h_c1, psi, constants)
}

/// Hyperfine term c3c.
pub fn h_c3c(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    let factor = -10.0 * constants.c3 / constants.c4 / constants.b_rot / ((2 * psi.j + 3) * (2 * psi.j - 1)) as f64;
    factor * com(h_c4, h_rot, psi, constants)
}

/// Rotational Hamiltonian: B * J^2
pub fn h_rot(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    constants.b_rot * j2(psi, constants)
}

/// Helper for c3 term.
pub fn h_i1r(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    com(i1z, r10, psi, constants) + (
        com(i1p, r1m, psi, constants) - com(i1m, r1p, psi, constants)
    ) / 2.0f64.sqrt()
}

/// Helper for c3 term.
pub fn h_i2r(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    com(i2z, r10, psi, constants) + (
        com(i2p, r1m, psi, constants) - com(i2m, r1p, psi, constants)
    ) / 2.0f64.sqrt()
}

/// Alternative c3 term implementation.
pub fn h_c3_alt(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    5.0 * constants.c3 / constants.c4 * h_c4(psi, constants)
    - 15.0 * constants.c3 / 2.0 * (
        com(h_i1r, h_i2r, psi, constants) + com(h_i2r, h_i1r, psi, constants)
    )
}

/// Field-free Hamiltonian.
pub fn h_ff(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    h_rot(psi, constants)
    + h_c1(psi, constants)
    + h_c2(psi, constants)
    + h_c3_alt(psi, constants)
    + h_c4(psi, constants)
}

/// Zeeman Hamiltonian for B along x.
pub fn h_zx(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    if psi.j != 0 {
        -constants.mu_j / (psi.j as f64) * jx(psi, constants)
        - constants.mu_tl / (psi.i1 as f64 / 2.0) * i1x(psi, constants)
        - constants.mu_f / (psi.i2 as f64 / 2.0) * i2x(psi, constants)
    } else {
        -constants.mu_tl / (psi.i1 as f64 / 2.0) * i1x(psi, constants)
        - constants.mu_f / (psi.i2 as f64 / 2.0) * i2x(psi, constants)
    }
}

/// Zeeman Hamiltonian for B along y.
pub fn h_zy(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    if psi.j != 0 {
        -constants.mu_j / (psi.j as f64) * jy(psi, constants)
        - constants.mu_tl / (psi.i1 as f64 / 2.0) * i1y(psi, constants)
        - constants.mu_f / (psi.i2 as f64 / 2.0) * i2y(psi, constants)
    } else {
        -constants.mu_tl / (psi.i1 as f64 / 2.0) * i1y(psi, constants)
        - constants.mu_f / (psi.i2 as f64 / 2.0) * i2y(psi, constants)
    }
}

/// Zeeman Hamiltonian for B along z.
pub fn h_zz(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    if psi.j != 0 {
        -constants.mu_j / (psi.j as f64) * jz(psi, constants)
        - constants.mu_tl / (psi.i1 as f64 / 2.0) * i1z(psi, constants)
        - constants.mu_f / (psi.i2 as f64 / 2.0) * i2z(psi, constants)
    } else {
        -constants.mu_tl / (psi.i1 as f64 / 2.0) * i1z(psi, constants)
        - constants.mu_f / (psi.i2 as f64 / 2.0) * i2z(psi, constants)
    }
}

/// Spherical tensor component R_1^0.
pub fn r10(psi: UncoupledBasisState, _constants: &XConstants) -> UncoupledState {
    let j = psi.j as f64;
    let mj = psi.mj as f64;

    let amp1 = 2.0f64.sqrt() * ((j - mj) * (j + mj) / (8.0 * j * j - 2.0)).sqrt();
    let mut ket1 = psi;
    ket1.j -= 1;
    ket1.parity = parity_x(ket1.j);

    let amp2 = 2.0f64.sqrt() * ((j - mj + 1.0) * (j + mj + 1.0) / (8.0 * j * j + 16.0 * j + 6.0)).sqrt();
    let mut ket2 = psi;
    ket2.j += 1;
    ket2.parity = parity_x(ket2.j);

    UncoupledState {
        terms: vec![
            (Complex64::new(amp1, 0.0), ket1),
            (Complex64::new(amp2, 0.0), ket2)
        ]
    }
}

/// Spherical tensor component R_1^-1.
pub fn r1m(psi: UncoupledBasisState, _constants: &XConstants) -> UncoupledState {
    let j = psi.j as f64;
    let mj = psi.mj as f64;

    let amp1 = -0.5 * 2.0f64.sqrt() * ((j + mj) * (j + mj - 1.0) / (4.0 * j * j - 1.0)).sqrt();
    let mut ket1 = psi;
    ket1.j -= 1;
    ket1.mj -= 1;
    ket1.parity = parity_x(ket1.j);

    let amp2 = 0.5 * 2.0f64.sqrt() * ((j - mj + 1.0) * (j - mj + 2.0) / (3.0 + 4.0 * j * (j + 2.0))).sqrt();
    let mut ket2 = psi;
    ket2.j += 1;
    ket2.mj -= 1;
    ket2.parity = parity_x(ket2.j);

    UncoupledState {
        terms: vec![
            (Complex64::new(amp1, 0.0), ket1),
            (Complex64::new(amp2, 0.0), ket2)
        ]
    }
}

/// Spherical tensor component R_1^+1.
pub fn r1p(psi: UncoupledBasisState, _constants: &XConstants) -> UncoupledState {
    let j = psi.j as f64;
    let mj = psi.mj as f64;

    let amp1 = -0.5 * 2.0f64.sqrt() * ((j - mj) * (j - mj - 1.0) / (4.0 * j * j - 1.0)).sqrt();
    let mut ket1 = psi;
    ket1.j -= 1;
    ket1.mj += 1;
    ket1.parity = parity_x(ket1.j);

    let amp2 = 0.5 * 2.0f64.sqrt() * ((j + mj + 1.0) * (j + mj + 2.0) / (3.0 + 4.0 * j * (j + 2.0))).sqrt();
    let mut ket2 = psi;
    ket2.j += 1;
    ket2.mj += 1;
    ket2.parity = parity_x(ket2.j);

    UncoupledState {
        terms: vec![
            (Complex64::new(amp1, 0.0), ket1),
            (Complex64::new(amp2, 0.0), ket2)
        ]
    }
}

/// Stark Hamiltonian for E along x.
pub fn h_sx(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    -constants.d_tlf * (r1m(psi, constants) - r1p(psi, constants)) / 2.0f64.sqrt()
}

/// Stark Hamiltonian for E along y.
pub fn h_sy(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    let imag_unit = Complex64::new(0.0, 1.0);
    -constants.d_tlf * imag_unit * (r1m(psi, constants) + r1p(psi, constants)) / 2.0f64.sqrt()
}

/// Stark Hamiltonian for E along z.
pub fn h_sz(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    -constants.d_tlf * r10(psi, constants)
}
