use crate::states::{UncoupledBasisState, UncoupledState};
use num_complex::Complex64;

/// J^2 operator.
pub fn j2<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let j = psi.j as f64;
    let eigen = j * (j + 1.0);

    UncoupledState {
        terms: vec![(Complex64::new(eigen, 0.0), psi)],
    }
}

/// J^4 operator.
pub fn j4<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let j = psi.j;
    let j2_value = (j * (j + 1)) as f64;
    let j4_value = j2_value * j2_value;

    UncoupledState {
        terms: vec![(Complex64::new(j4_value, 0.0), psi)],
    }
}

/// J^6 operator.
pub fn j6<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let j = psi.j;
    let j2_value = (j * (j + 1)) as f64;
    let j6_value = j2_value * j2_value * j2_value;

    UncoupledState {
        terms: vec![(Complex64::new(j6_value, 0.0), psi)],
    }
}

/// J_z operator.
pub fn jz<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let mj = psi.mj as f64;
    UncoupledState {
        terms: vec![(Complex64::new(mj, 0.0), psi)],
    }
}

/// I1_z operator.
pub fn i1z<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let m1 = psi.m1 as f64 / 2.0;
    UncoupledState {
        terms: vec![(Complex64::new(m1, 0.0), psi)],
    }
}

/// I2_z operator.
pub fn i2z<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let m2 = psi.m2 as f64 / 2.0;
    UncoupledState {
        terms: vec![(Complex64::new(m2, 0.0), psi)],
    }
}

/// J_+ operator.
pub fn jp<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let j = psi.j;
    let mj = psi.mj;

    if (mj as f64) < (j as f64) {
        let coeff = ((j as f64 * (j as f64 + 1.0) - mj as f64 * (mj as f64 + 1.0))).sqrt();
        let mut new_psi = psi;
        new_psi.mj += 1;
        UncoupledState {
            terms: vec![(Complex64::new(coeff, 0.0), new_psi)],
        }
    } else {
        UncoupledState { terms: vec![] }
    }
}

/// J_- operator.
pub fn jm<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let j = psi.j;
    let mj = psi.mj;

    if (mj as f64) > -(j as f64) {
        let coeff = ((j as f64 + mj as f64) * (j as f64 - mj as f64 + 1.0)).sqrt();
        let mut new_psi = psi;
        new_psi.mj -= 1;
        UncoupledState {
            terms: vec![(Complex64::new(coeff, 0.0), new_psi)],
        }
    } else {
        UncoupledState { terms: vec![] }
    }
}

/// I1_+ operator.
pub fn i1p<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let i1 = psi.i1;
    let m1 = psi.m1;

    if m1 < i1 {
        let coeff = 0.5 * ((i1 - m1) as f64 * (i1 + m1 + 2) as f64).sqrt();
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
pub fn i1m<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let i1 = psi.i1;
    let m1 = psi.m1;

    if m1 > -i1 {
        let coeff = 0.5 * ((i1 + m1) as f64 * (i1 - m1 + 2) as f64).sqrt();
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
pub fn i2p<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let i2 = psi.i2;
    let m2 = psi.m2;

    if m2 < i2 {
        let coeff = 0.5 * ((i2 - m2) as f64 * (i2 + m2 + 2) as f64).sqrt();
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
pub fn i2m<T>(psi: UncoupledBasisState, _: &T) -> UncoupledState {
    let i2 = psi.i2;
    let m2 = psi.m2;

    if m2 > -i2 {
        let coeff = 0.5 * ((i2 + m2) as f64 * (i2 - m2 + 2) as f64).sqrt();
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
pub fn jx<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    0.5 * (jp(psi, constants) + jm(psi, constants))
}

/// J_y operator.
pub fn jy<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    let imag_unit = num_complex::Complex64::new(0.0, 1.0);
    (0.0 - imag_unit) * 0.5 * (jp(psi, constants) - jm(psi, constants))
}

/// I1_x operator.
pub fn i1x<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    0.5 * (i1p(psi, constants) + i1m(psi, constants))
}

/// I1_y operator.
pub fn i1y<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    let imag_unit = num_complex::Complex64::new(0.0, 1.0);
    (0.0 - imag_unit) * 0.5 * (i1p(psi, constants) - i1m(psi, constants))
}

/// I2_x operator.
pub fn i2x<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    0.5 * (i2p(psi, constants) + i2m(psi, constants))
}

/// I2_y operator.
pub fn i2y<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    let imag_unit = num_complex::Complex64::new(0.0, 1.0);
    (0.0 - imag_unit) * 0.5 * (i2p(psi, constants) - i2m(psi, constants))
}

/// Commutator helper: [A, B] applied to psi.
/// Actually computes A(B(psi)) which is just composition, not commutator?
/// Wait, the implementation is `result = result + A(basis, constants) * amp`.
/// This is applying A after B. So it's A * B.
/// The name `com` suggests commutator or composition.
/// Looking at usage in `X_uncoupled.rs`: `com(I1z, Jz, psi, constants)`
/// This is used for terms like `I1z * Jz`. So it is composition.
pub fn com<T>(
    a: fn(UncoupledBasisState, &T) -> UncoupledState,
    b: fn(UncoupledBasisState, &T) -> UncoupledState,
    psi: UncoupledBasisState,
    constants: &T
) -> UncoupledState {
    let intermediate = b(psi, constants);
    let mut result = UncoupledState::empty();
    for (amp, basis) in intermediate.terms {
        result = result + a(basis, constants) * amp;
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
