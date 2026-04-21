use crate::b_coupled;
use crate::constants::{BConstants, XConstants};
use crate::states::{CoupledBasisState, CoupledState, UncoupledBasisState, UncoupledState};
use crate::x_uncoupled;
use num_complex::Complex64;
use rayon::prelude::*;

pub struct HamiltonianUncoupledX {
    pub h_ff: Vec<Complex64>,
    pub h_sx: Vec<Complex64>,
    pub h_sy: Vec<Complex64>,
    pub h_sz: Vec<Complex64>,
    pub h_zx: Vec<Complex64>,
    pub h_zy: Vec<Complex64>,
    pub h_zz: Vec<Complex64>,
}

pub struct HamiltonianCoupledB {
    pub h_rot: Vec<Complex64>,
    pub h_mhf_tl: Vec<Complex64>,
    pub h_mhf_f: Vec<Complex64>,
    pub h_ld: Vec<Complex64>,
    pub h_cp1_tl: Vec<Complex64>,
    pub h_c_tl: Vec<Complex64>,
    pub h_sx: Vec<Complex64>,
    pub h_sy: Vec<Complex64>,
    pub h_sz: Vec<Complex64>,
    pub h_zx: Vec<Complex64>,
    pub h_zy: Vec<Complex64>,
    pub h_zz: Vec<Complex64>,
}

pub trait OperatorState: Sized {
    type BasisState: Copy + Eq;
    fn terms(&self) -> &[(Complex64, Self::BasisState)];
}

impl OperatorState for UncoupledState {
    type BasisState = UncoupledBasisState;
    #[inline]
    fn terms(&self) -> &[(Complex64, UncoupledBasisState)] {
        &self.terms
    }
}

impl OperatorState for CoupledState {
    type BasisState = CoupledBasisState;
    #[inline]
    fn terms(&self) -> &[(Complex64, CoupledBasisState)] {
        &self.terms
    }
}

fn h_mat_elems_generic<B, S, C>(h: fn(B, &C) -> S, qn: &[B], constants: &C) -> Vec<Complex64>
where
    B: Copy + Eq,
    S: OperatorState<BasisState = B>,
{
    let n = qn.len();
    let mut result = vec![Complex64::ZERO; n * n];

    let h_applied: Vec<S> = qn.iter().map(|b| h(*b, constants)).collect();

    for (i, a) in qn.iter().enumerate() {
        for j in i..n {
            let psi = &h_applied[j];
            let mut val = Complex64::ZERO;
            for &(amp, ref basis) in psi.terms() {
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

#[inline]
pub fn h_mat_elems(
    h: fn(UncoupledBasisState, &XConstants) -> UncoupledState,
    qn: &[UncoupledBasisState],
    constants: &XConstants,
) -> Vec<Complex64> {
    h_mat_elems_generic(h, qn, constants)
}

#[inline]
pub fn h_mat_elems_b(
    h: fn(CoupledBasisState, &BConstants) -> CoupledState,
    qn: &[CoupledBasisState],
    constants: &BConstants,
) -> Vec<Complex64> {
    h_mat_elems_generic(h, qn, constants)
}

pub fn generate_uncoupled_hamiltonian_x(
    qn: &[UncoupledBasisState],
    constants: &XConstants,
) -> HamiltonianUncoupledX {
    let ops: Vec<fn(UncoupledBasisState, &XConstants) -> UncoupledState> = vec![
        x_uncoupled::h_ff,
        x_uncoupled::h_sx,
        x_uncoupled::h_sy,
        x_uncoupled::h_sz,
        x_uncoupled::h_zx,
        x_uncoupled::h_zy,
        x_uncoupled::h_zz,
    ];

    let results: Vec<Vec<Complex64>> = ops
        .into_par_iter()
        .map(|op| h_mat_elems(op, qn, constants))
        .collect();

    let mut it = results.into_iter();
    HamiltonianUncoupledX {
        h_ff: it.next().unwrap(),
        h_sx: it.next().unwrap(),
        h_sy: it.next().unwrap(),
        h_sz: it.next().unwrap(),
        h_zx: it.next().unwrap(),
        h_zy: it.next().unwrap(),
        h_zz: it.next().unwrap(),
    }
}

pub fn generate_coupled_hamiltonian_b(
    qn: &[CoupledBasisState],
    constants: &BConstants,
) -> HamiltonianCoupledB {
    let ops: Vec<fn(CoupledBasisState, &BConstants) -> CoupledState> = vec![
        b_coupled::h_rot,
        b_coupled::h_mhf_tl,
        b_coupled::h_mhf_f,
        b_coupled::h_ld,
        b_coupled::h_cp1_tl,
        b_coupled::h_c_tl,
        b_coupled::h_sx,
        b_coupled::h_sy,
        b_coupled::h_sz,
        b_coupled::h_zx,
        b_coupled::h_zy,
        b_coupled::h_zz,
    ];

    let results: Vec<Vec<Complex64>> = ops
        .into_par_iter()
        .map(|op| h_mat_elems_b(op, qn, constants))
        .collect();

    let mut it = results.into_iter();
    HamiltonianCoupledB {
        h_rot: it.next().unwrap(),
        h_mhf_tl: it.next().unwrap(),
        h_mhf_f: it.next().unwrap(),
        h_ld: it.next().unwrap(),
        h_cp1_tl: it.next().unwrap(),
        h_c_tl: it.next().unwrap(),
        h_sx: it.next().unwrap(),
        h_sy: it.next().unwrap(),
        h_sz: it.next().unwrap(),
        h_zx: it.next().unwrap(),
        h_zy: it.next().unwrap(),
        h_zz: it.next().unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::XConstants;
    use crate::states::ElectronicState;
    use crate::x_uncoupled;

    fn make_x_basis(j: i32) -> Vec<UncoupledBasisState> {
        let i1 = 1;
        let i2 = 1;
        let mut states = Vec::new();
        for mj in (-j..=j).step_by(2) {
            for m1 in (-i1..=i1).step_by(2) {
                for m2 in (-i2..=i2).step_by(2) {
                    states.push(UncoupledBasisState {
                        j,
                        mj,
                        i1,
                        m1,
                        i2,
                        m2,
                        omega: 0,
                        parity: 1,
                    });
                }
            }
        }
        states
    }

    #[test]
    fn test_h_mat_elems_hermitian() {
        let qn = make_x_basis(2);
        let constants = XConstants::default();
        let mat = h_mat_elems(x_uncoupled::h_ff, &qn, &constants);
        let n = qn.len();
        for i in 0..n {
            for j in 0..n {
                let diff = (mat[i * n + j] - mat[j * n + i].conj()).norm();
                assert!(diff < 1e-12, "not Hermitian at ({i},{j}): diff={diff}");
            }
        }
    }

    #[test]
    fn test_h_mat_elems_diagonal_real() {
        let qn = make_x_basis(2);
        let constants = XConstants::default();
        let mat = h_mat_elems(x_uncoupled::h_ff, &qn, &constants);
        let n = qn.len();
        for i in 0..n {
            assert!(
                mat[i * n + i].im.abs() < 1e-14,
                "diagonal element ({i},{i}) has nonzero imaginary part: {}",
                mat[i * n + i].im
            );
        }
    }

    #[test]
    fn test_h_mat_elems_b_hermitian() {
        let states = vec![
            CoupledBasisState {
                j: 1,
                f: 1,
                mf: 0,
                i1: 1,
                i2: 1,
                f1: 1,
                omega: 1,
                parity: None,
                electronic_state: ElectronicState::B,
            },
            CoupledBasisState {
                j: 1,
                f: 1,
                mf: 0,
                i1: 1,
                i2: 1,
                f1: 3,
                omega: 1,
                parity: None,
                electronic_state: ElectronicState::B,
            },
            CoupledBasisState {
                j: 1,
                f: 2,
                mf: 0,
                i1: 1,
                i2: 1,
                f1: 3,
                omega: 1,
                parity: None,
                electronic_state: ElectronicState::B,
            },
        ];
        let constants = BConstants::default();
        let mat = h_mat_elems_b(b_coupled::h_mhf_tl, &states, &constants);
        let n = states.len();
        for i in 0..n {
            for j in 0..n {
                let diff = (mat[i * n + j] - mat[j * n + i].conj()).norm();
                assert!(diff < 1e-12, "not Hermitian at ({i},{j}): diff={diff}");
            }
        }
    }

    #[test]
    fn test_h_mat_elems_correct_size() {
        let qn = make_x_basis(0);
        let constants = XConstants::default();
        let mat = h_mat_elems(x_uncoupled::h_ff, &qn, &constants);
        assert_eq!(mat.len(), qn.len() * qn.len());
    }

    #[test]
    fn test_generate_uncoupled_hamiltonian_x_produces_all_components() {
        let qn = make_x_basis(0);
        let constants = XConstants::default();
        let ham = generate_uncoupled_hamiltonian_x(&qn, &constants);
        let n2 = qn.len() * qn.len();
        assert_eq!(ham.h_ff.len(), n2);
        assert_eq!(ham.h_sx.len(), n2);
        assert_eq!(ham.h_sy.len(), n2);
        assert_eq!(ham.h_sz.len(), n2);
        assert_eq!(ham.h_zx.len(), n2);
        assert_eq!(ham.h_zy.len(), n2);
        assert_eq!(ham.h_zz.len(), n2);
    }
}
