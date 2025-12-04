use crate::constants::XConstants;
use crate::states::{UncoupledBasisState, UncoupledState};
use crate::quantum_operators::*;
use num_complex::Complex64;

/// Calculate parity of X state.
pub fn parity_X(J: i32) -> i8 {
    if J % 2 == 0 { 1 } else { -1 }
}

/// Hyperfine term c1: c1 * I1 . J
pub fn Hc1(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    constants.c1 * (
        com(I1z, Jz, psi, constants)
        + 0.5 * (com(I1p, Jm, psi, constants) + com(I1m, Jp, psi, constants))
    )
}

/// Hyperfine term c2: c2 * I2 . J
pub fn Hc2(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    constants.c2 * (
        com(I2z, Jz, psi, constants)
        + 0.5 * (com(I2p, Jm, psi, constants) + com(I2m, Jp, psi, constants))
    )
}

/// Hyperfine term c4: c4 * I1 . I2
pub fn Hc4(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    constants.c4 * (
        com(I1z, I2z, psi, constants)
        + 0.5 * (com(I1p, I2m, psi, constants) + com(I1m, I2p, psi, constants))
    )
}

/// Hyperfine term c3a.
pub fn Hc3a(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    let factor = 15.0 * constants.c3 / constants.c1 / constants.c2 / ((2 * psi.J + 3) * (2 * psi.J - 1)) as f64;
    factor * com(Hc1, Hc2, psi, constants)
}

/// Hyperfine term c3b.
pub fn Hc3b(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    let factor = 15.0 * constants.c3 / constants.c2 / constants.c1 / ((2 * psi.J + 3) * (2 * psi.J - 1)) as f64;
    factor * com(Hc2, Hc1, psi, constants)
}

/// Hyperfine term c3c.
pub fn Hc3c(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    let factor = -10.0 * constants.c3 / constants.c4 / constants.B_rot / ((2 * psi.J + 3) * (2 * psi.J - 1)) as f64;
    factor * com(Hc4, Hrot, psi, constants)
}

/// Rotational Hamiltonian: B * J^2
pub fn Hrot(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    constants.B_rot * J2(psi, constants)
}

/// Helper for c3 term.
pub fn HI1R(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    com(I1z, R10, psi, constants) + (
        com(I1p, R1m, psi, constants) - com(I1m, R1p, psi, constants)
    ) / 2.0f64.sqrt()
}

/// Helper for c3 term.
pub fn HI2R(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    com(I2z, R10, psi, constants) + (
        com(I2p, R1m, psi, constants) - com(I2m, R1p, psi, constants)
    ) / 2.0f64.sqrt()
}

/// Alternative c3 term implementation.
pub fn Hc3_alt(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    5.0 * constants.c3 / constants.c4 * Hc4(psi, constants)
    - 15.0 * constants.c3 / 2.0 * (
        com(HI1R, HI2R, psi, constants) + com(HI2R, HI1R, psi, constants)
    )
}

/// Field-free Hamiltonian.
pub fn Hff(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    Hrot(psi, constants)
    + Hc1(psi, constants)
    + Hc2(psi, constants)
    + Hc3_alt(psi, constants)
    + Hc4(psi, constants)
}

/// Zeeman Hamiltonian for B along x.
pub fn HZx(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    if psi.J != 0 {
        -constants.mu_J / (psi.J as f64) * Jx(psi, constants)
        - constants.mu_Tl / (psi.I1 as f64 / 2.0) * I1x(psi, constants)
        - constants.mu_F / (psi.I2 as f64 / 2.0) * I2x(psi, constants)
    } else {
        -constants.mu_Tl / (psi.I1 as f64 / 2.0) * I1x(psi, constants)
        - constants.mu_F / (psi.I2 as f64 / 2.0) * I2x(psi, constants)
    }
}

/// Zeeman Hamiltonian for B along y.
pub fn HZy(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    if psi.J != 0 {
        -constants.mu_J / (psi.J as f64) * Jy(psi, constants)
        - constants.mu_Tl / (psi.I1 as f64 / 2.0) * I1y(psi, constants)
        - constants.mu_F / (psi.I2 as f64 / 2.0) * I2y(psi, constants)
    } else {
        -constants.mu_Tl / (psi.I1 as f64 / 2.0) * I1y(psi, constants)
        - constants.mu_F / (psi.I2 as f64 / 2.0) * I2y(psi, constants)
    }
}

/// Zeeman Hamiltonian for B along z.
pub fn HZz(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    if psi.J != 0 {
        -constants.mu_J / (psi.J as f64) * Jz(psi, constants)
        - constants.mu_Tl / (psi.I1 as f64 / 2.0) * I1z(psi, constants)
        - constants.mu_F / (psi.I2 as f64 / 2.0) * I2z(psi, constants)
    } else {
        -constants.mu_Tl / (psi.I1 as f64 / 2.0) * I1z(psi, constants)
        - constants.mu_F / (psi.I2 as f64 / 2.0) * I2z(psi, constants)
    }
}

/// Spherical tensor component R_1^0.
pub fn R10(psi: UncoupledBasisState, _constants: &XConstants) -> UncoupledState {
    let J = psi.J as f64;
    let mJ = psi.mJ as f64;

    let amp1 = 2.0f64.sqrt() * ((J - mJ) * (J + mJ) / (8.0 * J * J - 2.0)).sqrt();
    let mut ket1 = psi;
    ket1.J -= 1;
    ket1.parity = parity_X(ket1.J);

    let amp2 = 2.0f64.sqrt() * ((J - mJ + 1.0) * (J + mJ + 1.0) / (8.0 * J * J + 16.0 * J + 6.0)).sqrt();
    let mut ket2 = psi;
    ket2.J += 1;
    ket2.parity = parity_X(ket2.J);

    UncoupledState {
        terms: vec![
            (Complex64::new(amp1, 0.0), ket1),
            (Complex64::new(amp2, 0.0), ket2)
        ]
    }
}

/// Spherical tensor component R_1^-1.
pub fn R1m(psi: UncoupledBasisState, _constants: &XConstants) -> UncoupledState {
    let J = psi.J as f64;
    let mJ = psi.mJ as f64;

    let amp1 = -0.5 * 2.0f64.sqrt() * ((J + mJ) * (J + mJ - 1.0) / (4.0 * J * J - 1.0)).sqrt();
    let mut ket1 = psi;
    ket1.J -= 1;
    ket1.mJ -= 1;
    ket1.parity = parity_X(ket1.J);

    let amp2 = 0.5 * 2.0f64.sqrt() * ((J - mJ + 1.0) * (J - mJ + 2.0) / (3.0 + 4.0 * J * (J + 2.0))).sqrt();
    let mut ket2 = psi;
    ket2.J += 1;
    ket2.mJ -= 1;
    ket2.parity = parity_X(ket2.J);

    UncoupledState {
        terms: vec![
            (Complex64::new(amp1, 0.0), ket1),
            (Complex64::new(amp2, 0.0), ket2)
        ]
    }
}

/// Spherical tensor component R_1^+1.
pub fn R1p(psi: UncoupledBasisState, _constants: &XConstants) -> UncoupledState {
    let J = psi.J as f64;
    let mJ = psi.mJ as f64;

    let amp1 = -0.5 * 2.0f64.sqrt() * ((J - mJ) * (J - mJ - 1.0) / (4.0 * J * J - 1.0)).sqrt();
    let mut ket1 = psi;
    ket1.J -= 1;
    ket1.mJ += 1;
    ket1.parity = parity_X(ket1.J);

    let amp2 = 0.5 * 2.0f64.sqrt() * ((J + mJ + 1.0) * (J + mJ + 2.0) / (3.0 + 4.0 * J * (J + 2.0))).sqrt();
    let mut ket2 = psi;
    ket2.J += 1;
    ket2.mJ += 1;
    ket2.parity = parity_X(ket2.J);

    UncoupledState {
        terms: vec![
            (Complex64::new(amp1, 0.0), ket1),
            (Complex64::new(amp2, 0.0), ket2)
        ]
    }
}

/// Stark Hamiltonian for E along x.
pub fn HSx(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    -constants.D_TlF * (R1m(psi, constants) - R1p(psi, constants)) / 2.0f64.sqrt()
}

/// Stark Hamiltonian for E along y.
pub fn HSy(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    let imag_unit = Complex64::new(0.0, 1.0);
    -constants.D_TlF * imag_unit * (R1m(psi, constants) + R1p(psi, constants)) / 2.0f64.sqrt()
}

/// Stark Hamiltonian for E along z.
pub fn HSz(psi: UncoupledBasisState, constants: &XConstants) -> UncoupledState {
    -constants.D_TlF * R10(psi, constants)
}
