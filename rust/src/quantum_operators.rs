use crate::states::{UncoupledBasisState, UncoupledState};
use num_complex::Complex64;

/// J^2 operator.
pub fn J2<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let J = psi.J as f64;
    let eigen = J * (J + 1.0);

    UncoupledState {
        terms: vec![(Complex64::new(eigen, 0.0), psi)],
    }
}

/// J^4 operator.
pub fn J4<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let J = psi.J;
    let J2_value = (J * (J + 1)) as f64;
    let J4_value = J2_value * J2_value;

    UncoupledState {
        terms: vec![(Complex64::new(J4_value, 0.0), psi)],
    }
}

/// J^6 operator.
pub fn J6<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let J = psi.J;
    let J2_value = (J * (J + 1)) as f64;
    let J6_value = J2_value * J2_value * J2_value;

    UncoupledState {
        terms: vec![(Complex64::new(J6_value, 0.0), psi)],
    }
}

/// J_z operator.
pub fn Jz<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let mJ = psi.mJ as f64;
    UncoupledState {
        terms: vec![(Complex64::new(mJ, 0.0), psi)],
    }
}

/// I1_z operator.
pub fn I1z<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let m1 = psi.m1 as f64 / 2.0;
    UncoupledState {
        terms: vec![(Complex64::new(m1, 0.0), psi)],
    }
}

/// I2_z operator.
pub fn I2z<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let m2 = psi.m2 as f64 / 2.0;
    UncoupledState {
        terms: vec![(Complex64::new(m2, 0.0), psi)],
    }
}

/// J_+ operator.
pub fn Jp<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let J = psi.J;
    let mJ = psi.mJ;

    if (mJ as f64) < (J as f64) {
        let coeff = ((J as f64 * (J as f64 + 1.0) - mJ as f64 * (mJ as f64 + 1.0))).sqrt();
        let mut new_psi = psi;
        new_psi.mJ += 1;
        UncoupledState {
            terms: vec![(Complex64::new(coeff, 0.0), new_psi)],
        }
    } else {
        UncoupledState { terms: vec![] }
    }
}

/// J_- operator.
pub fn Jm<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let J = psi.J;
    let mJ = psi.mJ;

    if (mJ as f64) > -(J as f64) {
        let coeff = ((J as f64 + mJ as f64) * (J as f64 - mJ as f64 + 1.0)).sqrt();
        let mut new_psi = psi;
        new_psi.mJ -= 1;
        UncoupledState {
            terms: vec![(Complex64::new(coeff, 0.0), new_psi)],
        }
    } else {
        UncoupledState { terms: vec![] }
    }
}

/// I1_+ operator.
pub fn I1p<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let I1 = psi.I1;
    let m1 = psi.m1;

    if m1 < I1 {
        let coeff = 0.5 * ((I1 - m1) as f64 * (I1 + m1 + 2) as f64).sqrt();
        let mut new_psi = psi;
        new_psi.m1 += 2;
        UncoupledState {
            terms: vec![(Complex64::new(coeff, 0.0), new_psi)],
        }
    } else {
        UncoupledState { terms: vec![] }
    }
}

/// I1_- operator.
pub fn I1m<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let I1 = psi.I1;
    let m1 = psi.m1;

    if m1 > -I1 {
        let coeff = 0.5 * ((I1 + m1) as f64 * (I1 - m1 + 2) as f64).sqrt();
        let mut new_psi = psi;
        new_psi.m1 -= 2;
        UncoupledState {
            terms: vec![(Complex64::new(coeff, 0.0), new_psi)],
        }
    } else {
        UncoupledState { terms: vec![] }
    }
}

/// I2_+ operator.
pub fn I2p<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let I2 = psi.I2;
    let m2 = psi.m2;

    if m2 < I2 {
        let coeff = 0.5 * ((I2 - m2) as f64 * (I2 + m2 + 2) as f64).sqrt();
        let mut new_psi = psi;
        new_psi.m2 += 2;
        UncoupledState {
            terms: vec![(Complex64::new(coeff, 0.0), new_psi)],
        }
    } else {
        UncoupledState { terms: vec![] }
    }
}

/// I2_- operator.
pub fn I2m<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let I2 = psi.I2;
    let m2 = psi.m2;

    if m2 > -I2 {
        let coeff = 0.5 * ((I2 + m2) as f64 * (I2 - m2 + 2) as f64).sqrt();
        let mut new_psi = psi;
        new_psi.m2 -= 2;
        UncoupledState {
            terms: vec![(Complex64::new(coeff, 0.0), new_psi)],
        }
    } else {
        UncoupledState { terms: vec![] }
    }
}

/// J_x operator.
pub fn Jx<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    0.5 * (Jp(psi, constants) + Jm(psi, constants))
}

/// J_y operator.
pub fn Jy<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    let imag_unit = num_complex::Complex64::new(0.0, 1.0);
    (0.0 - imag_unit) * 0.5 * (Jp(psi, constants) - Jm(psi, constants))
}

/// I1_x operator.
pub fn I1x<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    0.5 * (I1p(psi, constants) + I1m(psi, constants))
}

/// I1_y operator.
pub fn I1y<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    let imag_unit = num_complex::Complex64::new(0.0, 1.0);
    (0.0 - imag_unit) * 0.5 * (I1p(psi, constants) - I1m(psi, constants))
}

/// I2_x operator.
pub fn I2x<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    0.5 * (I2p(psi, constants) + I2m(psi, constants))
}

/// I2_y operator.
pub fn I2y<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    let imag_unit = num_complex::Complex64::new(0.0, 1.0);
    (0.0 - imag_unit) * 0.5 * (I2p(psi, constants) - I2m(psi, constants))
}

/// Commutator helper: [A, B] applied to psi.
/// Actually computes A(B(psi)) which is just composition, not commutator?
/// Wait, the implementation is `result = result + A(basis, constants) * amp`.
/// This is applying A after B. So it's A * B.
/// The name `com` suggests commutator or composition.
/// Looking at usage in `X_uncoupled.rs`: `com(I1z, Jz, psi, constants)`
/// This is used for terms like `I1z * Jz`. So it is composition.
pub fn com<T>(
    A: fn(UncoupledBasisState, &T) -> UncoupledState,
    B: fn(UncoupledBasisState, &T) -> UncoupledState,
    psi: UncoupledBasisState,
    constants: &T
) -> UncoupledState {
    let intermediate = B(psi, constants);
    let mut result = UncoupledState::empty();
    for (amp, basis) in intermediate.terms {
        result = result + A(basis, constants) * amp;
    }
    result
}

/// Apply an operator to a state (superposition).
pub fn apply_op<T>(op: fn(UncoupledBasisState, &T) -> UncoupledState, state: UncoupledState, constants: &T) -> UncoupledState {
    let mut result = UncoupledState::empty();
    for (amp, basis) in state.terms {
        let new_state = op(basis, constants);
        result = result + new_state * amp;
    }
    result
}
