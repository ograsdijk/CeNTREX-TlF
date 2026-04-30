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

    if mj < j {
        let jf = j as f64;
        let mjf = mj as f64;
        let coeff = (jf * (jf + 1.0) - mjf * (mjf + 1.0)).sqrt();
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

    if mj > -j {
        let jf = j as f64;
        let mjf = mj as f64;
        let coeff = ((jf + mjf) * (jf - mjf + 1.0)).sqrt();
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
    let imag_unit = Complex64::I;
    (0.0 - imag_unit) * 0.5 * (jp(psi, constants) - jm(psi, constants))
}

/// I1_x operator.
pub fn i1x<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    0.5 * (i1p(psi, constants) + i1m(psi, constants))
}

/// I1_y operator.
pub fn i1y<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    let imag_unit = Complex64::I;
    (0.0 - imag_unit) * 0.5 * (i1p(psi, constants) - i1m(psi, constants))
}

/// I2_x operator.
pub fn i2x<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    0.5 * (i2p(psi, constants) + i2m(psi, constants))
}

/// I2_y operator.
pub fn i2y<T>(psi: UncoupledBasisState, constants: &T) -> UncoupledState {
    let imag_unit = Complex64::I;
    (0.0 - imag_unit) * 0.5 * (i2p(psi, constants) - i2m(psi, constants))
}

/// Operator composition: A(B(psi)).
pub fn com<T>(
    a: fn(UncoupledBasisState, &T) -> UncoupledState,
    b: fn(UncoupledBasisState, &T) -> UncoupledState,
    psi: UncoupledBasisState,
    constants: &T,
) -> UncoupledState {
    let intermediate = b(psi, constants);
    let mut result = UncoupledState::empty();
    for (amp, basis) in intermediate.terms {
        result = result + a(basis, constants) * amp;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_state(j: i32, mj: i32) -> UncoupledBasisState {
        UncoupledBasisState {
            j,
            mj,
            i1: 1,
            m1: 1,
            i2: 1,
            m2: 1,
            omega: 0,
            parity: 1,
        }
    }

    #[test]
    fn test_j2_eigenvalue() {
        let psi = make_state(2, 1);
        let result = j2(psi, &());
        assert_eq!(result.terms.len(), 1);
        assert!((result.terms[0].0.re - 6.0).abs() < 1e-14); // 2*(2+1)=6
    }

    #[test]
    fn test_j4_eigenvalue() {
        let psi = make_state(2, 0);
        let result = j4(psi, &());
        assert!((result.terms[0].0.re - 36.0).abs() < 1e-14); // 6^2=36
    }

    #[test]
    fn test_j6_eigenvalue() {
        let psi = make_state(2, 0);
        let result = j6(psi, &());
        assert!((result.terms[0].0.re - 216.0).abs() < 1e-14); // 6^3=216
    }

    #[test]
    fn test_jz_eigenvalue() {
        let psi = make_state(2, -1);
        let result = jz(psi, &());
        assert!((result.terms[0].0.re - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_jp_raises_mj() {
        let psi = make_state(1, 0);
        let result = jp(psi, &());
        assert_eq!(result.terms.len(), 1);
        assert_eq!(result.terms[0].1.mj, 1);
        // coeff = sqrt(j(j+1) - mj(mj+1)) = sqrt(2 - 0) = sqrt(2)
        assert!((result.terms[0].0.re - 2.0_f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn test_jp_at_max_mj_is_zero() {
        let psi = make_state(1, 1);
        let result = jp(psi, &());
        assert!(result.terms.is_empty());
    }

    #[test]
    fn test_jm_lowers_mj() {
        let psi = make_state(1, 0);
        let result = jm(psi, &());
        assert_eq!(result.terms.len(), 1);
        assert_eq!(result.terms[0].1.mj, -1);
        assert!((result.terms[0].0.re - 2.0_f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn test_jm_at_min_mj_is_zero() {
        let psi = make_state(1, -1);
        let result = jm(psi, &());
        assert!(result.terms.is_empty());
    }

    #[test]
    fn test_i1z_eigenvalue() {
        let mut psi = make_state(1, 0);
        psi.i1 = 1;
        psi.m1 = -1; // m1/2 = -0.5
        let result = i1z(psi, &());
        assert!((result.terms[0].0.re - (-0.5)).abs() < 1e-14);
    }

    #[test]
    fn test_i1p_raises_m1() {
        let mut psi = make_state(1, 0);
        psi.i1 = 1;
        psi.m1 = -1;
        let result = i1p(psi, &());
        assert_eq!(result.terms[0].1.m1, 1);
    }

    #[test]
    fn test_i1p_at_max_is_zero() {
        let mut psi = make_state(1, 0);
        psi.i1 = 1;
        psi.m1 = 1;
        let result = i1p(psi, &());
        assert!(result.terms.is_empty());
    }

    #[test]
    fn test_jx_is_half_jp_plus_jm() {
        let psi = make_state(1, 0);
        let jx_result = jx(psi, &());
        let manual = 0.5 * (jp(psi, &()) + jm(psi, &()));
        assert_eq!(jx_result.terms.len(), manual.terms.len());
        for (a, b) in jx_result.terms.iter().zip(manual.terms.iter()) {
            assert!((a.0 - b.0).norm() < 1e-14);
            assert_eq!(a.1, b.1);
        }
    }

    #[test]
    fn test_com_jz_jz_is_j2z() {
        let psi = make_state(2, 1);
        let result = com(jz, jz, psi, &());
        // Jz(Jz|psi>) = mj^2 |psi> = 1.0 |psi>
        assert_eq!(result.terms.len(), 1);
        assert!((result.terms[0].0.re - 1.0).abs() < 1e-14);
    }
}
