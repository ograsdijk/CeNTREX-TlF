use crate::wigner::clebsch_gordan;
use num_complex::Complex64;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Sub;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Uncoupled basis state for X state.
pub struct UncoupledBasisState {
    /// Rotational quantum number J
    pub j: i32,
    /// Projection of J on z-axis
    pub mj: i32,
    /// Nuclear spin I1 (Tl) * 2
    pub i1: i32, // 2 * I1
    /// Projection of I1 on z-axis * 2
    pub m1: i32, // 2 * m1
    /// Nuclear spin I2 (F) * 2
    pub i2: i32, // 2 * I2
    /// Projection of I2 on z-axis * 2
    pub m2: i32, // 2 * m2
    /// Omega quantum number
    pub omega: i32,
    /// Parity
    pub parity: i8,
}

pub type Amp = Complex64;
pub type Term = (Amp, UncoupledBasisState);

#[derive(Clone, Debug)]
/// Superposition of uncoupled basis states.
pub struct UncoupledState {
    pub terms: Vec<Term>,
}

impl UncoupledState {
    /// Create a new state from an iterator of terms.
    pub fn new<I>(iter: I, remove_zero_amp: bool) -> Self
    where
        I: IntoIterator<Item = Term>,
    {
        let mut terms: Vec<Term> = Vec::new();

        for (amp, ket) in iter {
            if remove_zero_amp && amp == Complex64::ZERO {
                continue;
            }
            if let Some(existing) = terms.iter_mut().find(|(_, k)| *k == ket) {
                existing.0 += amp;
            } else {
                terms.push((amp, ket));
            }
        }

        if remove_zero_amp {
            terms.retain(|(amp, _)| amp.re != 0.0 || amp.im != 0.0);
        }

        UncoupledState { terms }
    }

    /// Create an empty state.
    pub fn empty() -> Self {
        UncoupledState { terms: Vec::new() }
    }
}

macro_rules! impl_state_ops {
    ($State:ident) => {
        impl Add for $State {
            type Output = $State;

            fn add(self, other: $State) -> $State {
                let mut terms = self.terms;
                for (amp, ket) in other.terms {
                    if let Some(existing) = terms.iter_mut().find(|(_, k)| *k == ket) {
                        existing.0 += amp;
                    } else {
                        terms.push((amp, ket));
                    }
                }
                $State { terms }
            }
        }

        impl Sub for $State {
            type Output = $State;

            fn sub(self, other: $State) -> $State {
                self + (other * -1.0)
            }
        }

        impl Mul<f64> for $State {
            type Output = $State;

            fn mul(self, rhs: f64) -> $State {
                let terms = self
                    .terms
                    .into_iter()
                    .map(|(amp, ket)| (amp * rhs, ket))
                    .filter(|(amp, _)| amp.re != 0.0 || amp.im != 0.0)
                    .collect();
                $State { terms }
            }
        }

        impl Mul<Complex64> for $State {
            type Output = $State;

            fn mul(self, rhs: Complex64) -> $State {
                let terms = self
                    .terms
                    .into_iter()
                    .map(|(amp, ket)| (amp * rhs, ket))
                    .filter(|(amp, _)| amp.re != 0.0 || amp.im != 0.0)
                    .collect();
                $State { terms }
            }
        }

        impl Div<f64> for $State {
            type Output = $State;

            fn div(self, rhs: f64) -> $State {
                self * (1.0 / rhs)
            }
        }

        impl Mul<$State> for f64 {
            type Output = $State;

            fn mul(self, rhs: $State) -> $State {
                rhs * self
            }
        }

        impl Mul<$State> for Complex64 {
            type Output = $State;

            fn mul(self, rhs: $State) -> $State {
                rhs * self
            }
        }
    };
}

impl_state_ops!(UncoupledState);

impl Mul<UncoupledState> for UncoupledState {
    type Output = UncoupledState;

    fn mul(self, rhs: UncoupledState) -> UncoupledState {
        let mut terms: Vec<Term> = Vec::new();
        for (amp1, ket1) in self.terms {
            for (amp2, ket2) in &rhs.terms {
                if ket1 == *ket2 {
                    let product = amp1 * amp2;
                    if let Some(existing) = terms.iter_mut().find(|(_, k)| *k == ket1) {
                        existing.0 += product;
                    } else {
                        terms.push((product, ket1));
                    }
                }
            }
        }
        UncoupledState { terms }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Electronic state enum.
pub enum ElectronicState {
    X,
    B,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Coupled basis state for B state.
pub struct CoupledBasisState {
    /// Rotational quantum number J
    pub j: i32,
    /// Total angular momentum F
    pub f: i32,
    /// Projection of F on z-axis
    pub mf: i32,
    /// Nuclear spin I1 (Tl) * 2
    pub i1: i32, // 2*I1
    /// Nuclear spin I2 (F) * 2
    pub i2: i32, // 2*I2
    /// Intermediate angular momentum F1 * 2
    pub f1: i32, // 2*F1
    /// Omega quantum number
    pub omega: i32,
    /// Parity
    pub parity: Option<i8>,
    /// Electronic state
    pub electronic_state: ElectronicState,
}

pub type CoupledTerm = (Amp, CoupledBasisState);

#[derive(Clone, Debug, PartialEq)]
/// Superposition of coupled basis states.
pub struct CoupledState {
    pub terms: Vec<CoupledTerm>,
}

impl CoupledState {
    /// Create a new state from an iterator of terms.
    pub fn new<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = CoupledTerm>,
    {
        let mut terms: Vec<CoupledTerm> = Vec::new();

        for (amp, ket) in iter {
            if let Some(existing) = terms.iter_mut().find(|(_, k)| *k == ket) {
                existing.0 += amp;
            } else {
                terms.push((amp, ket));
            }
        }

        terms.retain(|(amp, _)| amp.re != 0.0 || amp.im != 0.0);

        CoupledState { terms }
    }

    pub fn from_vec(terms: Vec<CoupledTerm>) -> Self {
        Self::new(terms)
    }
}

impl_state_ops!(CoupledState);

impl CoupledBasisState {
    /// Transform coupled basis state to uncoupled basis.
    pub fn transform_to_uncoupled(&self) -> UncoupledState {
        let j = self.j;
        let f = self.f;
        let mf = self.mf;
        let i1 = self.i1;
        let i2 = self.i2;
        let f1 = self.f1;

        let mut terms = Vec::new();

        // mF1 ranges from -F1 to F1 in steps of 2
        for mf1 in (-f1..=f1).step_by(2) {
            // mJ ranges from -J to J in steps of 1 (since J is integer)
            for mj in -j..=j {
                // m1 ranges from -I1 to I1 in steps of 2
                for m1 in (-i1..=i1).step_by(2) {
                    // m2 ranges from -I2 to I2 in steps of 2
                    for m2 in (-i2..=i2).step_by(2) {
                        // Check angular momentum conservation
                        // mJ is single, m1, mF1 are doubled
                        if 2 * mj + m1 != mf1 {
                            continue;
                        }
                        // mF1, m2 are doubled, mF is single
                        if mf1 + m2 != 2 * mf {
                            continue;
                        }

                        // Convert single integer quantum numbers to doubled for CG calculation
                        let cg1 = clebsch_gordan(2 * j, 2 * mj, i1, m1, f1, mf1);
                        if cg1 == 0.0 {
                            continue;
                        }

                        let cg2 = clebsch_gordan(f1, mf1, i2, m2, 2 * f, 2 * mf);
                        if cg2 == 0.0 {
                            continue;
                        }

                        let amp = Complex64::new(cg1 * cg2, 0.0);
                        let state = UncoupledBasisState {
                            j,
                            mj,
                            i1,
                            m1,
                            i2,
                            m2,
                            omega: self.omega,
                            parity: self.parity.unwrap_or(0),
                        };
                        terms.push((amp, state));
                    }
                }
            }
        }
        UncoupledState::new(terms, true)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Enum wrapper for both basis types.
pub enum BasisStateEnum {
    Coupled(CoupledBasisState),
    Uncoupled(UncoupledBasisState),
}

impl BasisStateEnum {
    /// Calculate inner product <self|other>.
    pub fn inner_product(&self, other: &BasisStateEnum) -> Complex64 {
        match (self, other) {
            (BasisStateEnum::Coupled(a), BasisStateEnum::Coupled(b)) => {
                if a == b {
                    Complex64::ONE
                } else {
                    Complex64::ZERO
                }
            }
            (BasisStateEnum::Uncoupled(a), BasisStateEnum::Uncoupled(b)) => {
                if a == b {
                    Complex64::ONE
                } else {
                    Complex64::ZERO
                }
            }
            (BasisStateEnum::Uncoupled(a), BasisStateEnum::Coupled(b)) => {
                let uncoupled_b = b.transform_to_uncoupled();
                for (amp, state) in uncoupled_b.terms {
                    if state == *a {
                        return amp;
                    }
                }
                Complex64::ZERO
            }
            (BasisStateEnum::Coupled(a), BasisStateEnum::Uncoupled(b)) => {
                let uncoupled_a = a.transform_to_uncoupled();
                for (amp, state) in uncoupled_a.terms {
                    if state == *b {
                        return amp.conj();
                    }
                }
                Complex64::ZERO
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uncoupled(j: i32, mj: i32) -> UncoupledBasisState {
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
    fn test_uncoupled_state_add_combines_same_ket() {
        let a = UncoupledState {
            terms: vec![(Complex64::new(1.0, 0.0), make_uncoupled(1, 0))],
        };
        let b = UncoupledState {
            terms: vec![(Complex64::new(2.0, 0.0), make_uncoupled(1, 0))],
        };
        let sum = a + b;
        assert_eq!(sum.terms.len(), 1);
        assert!((sum.terms[0].0.re - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_uncoupled_state_add_keeps_different_kets() {
        let a = UncoupledState {
            terms: vec![(Complex64::new(1.0, 0.0), make_uncoupled(1, 0))],
        };
        let b = UncoupledState {
            terms: vec![(Complex64::new(2.0, 0.0), make_uncoupled(1, 1))],
        };
        let sum = a + b;
        assert_eq!(sum.terms.len(), 2);
    }

    #[test]
    fn test_uncoupled_state_sub() {
        let a = UncoupledState {
            terms: vec![(Complex64::new(3.0, 0.0), make_uncoupled(1, 0))],
        };
        let b = UncoupledState {
            terms: vec![(Complex64::new(1.0, 0.0), make_uncoupled(1, 0))],
        };
        let diff = a - b;
        assert_eq!(diff.terms.len(), 1);
        assert!((diff.terms[0].0.re - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_uncoupled_state_mul_scalar() {
        let a = UncoupledState {
            terms: vec![(Complex64::new(2.0, 0.0), make_uncoupled(1, 0))],
        };
        let scaled = a * 3.0;
        assert!((scaled.terms[0].0.re - 6.0).abs() < 1e-14);
    }

    #[test]
    fn test_uncoupled_state_mul_complex() {
        let a = UncoupledState {
            terms: vec![(Complex64::new(1.0, 0.0), make_uncoupled(1, 0))],
        };
        let i = Complex64::new(0.0, 1.0);
        let result = a * i;
        assert!(result.terms[0].0.re.abs() < 1e-14);
        assert!((result.terms[0].0.im - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_uncoupled_state_div() {
        let a = UncoupledState {
            terms: vec![(Complex64::new(6.0, 0.0), make_uncoupled(1, 0))],
        };
        let result = a / 2.0;
        assert!((result.terms[0].0.re - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_uncoupled_state_mul_zero_filters() {
        let a = UncoupledState {
            terms: vec![(Complex64::new(1.0, 0.0), make_uncoupled(1, 0))],
        };
        let result = a * 0.0;
        assert!(result.terms.is_empty());
    }

    #[test]
    fn test_uncoupled_state_new_merges_duplicates() {
        let terms = vec![
            (Complex64::new(1.0, 0.0), make_uncoupled(1, 0)),
            (Complex64::new(2.0, 0.0), make_uncoupled(1, 0)),
            (Complex64::new(3.0, 0.0), make_uncoupled(1, 1)),
        ];
        let state = UncoupledState::new(terms, false);
        assert_eq!(state.terms.len(), 2);
        let amp_0: f64 = state
            .terms
            .iter()
            .filter(|(_, k)| k.mj == 0)
            .map(|(a, _)| a.re)
            .sum();
        assert!((amp_0 - 3.0).abs() < 1e-14);
    }

    fn make_coupled(j: i32, f: i32, mf: i32) -> CoupledBasisState {
        CoupledBasisState {
            j,
            f,
            mf,
            i1: 1,
            i2: 1,
            f1: 1,
            omega: 1,
            parity: None,
            electronic_state: ElectronicState::B,
        }
    }

    #[test]
    fn test_coupled_state_add() {
        let a = CoupledState::new(vec![(Complex64::new(1.0, 0.0), make_coupled(1, 1, 0))]);
        let b = CoupledState::new(vec![(Complex64::new(2.0, 0.0), make_coupled(1, 1, 0))]);
        let sum = a + b;
        assert_eq!(sum.terms.len(), 1);
        assert!((sum.terms[0].0.re - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_coupled_state_mul_scalar() {
        let a = CoupledState::new(vec![(Complex64::new(2.0, 0.0), make_coupled(1, 1, 0))]);
        let scaled = a * 5.0;
        assert!((scaled.terms[0].0.re - 10.0).abs() < 1e-14);
    }

    #[test]
    fn test_coupled_state_sub() {
        let a = CoupledState::new(vec![(Complex64::new(5.0, 0.0), make_coupled(1, 1, 0))]);
        let b = CoupledState::new(vec![(Complex64::new(2.0, 0.0), make_coupled(1, 1, 0))]);
        let diff = a - b;
        assert_eq!(diff.terms.len(), 1);
        assert!((diff.terms[0].0.re - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_basis_state_enum_inner_product_same_uncoupled() {
        let a = BasisStateEnum::Uncoupled(make_uncoupled(1, 0));
        let b = BasisStateEnum::Uncoupled(make_uncoupled(1, 0));
        let ip = a.inner_product(&b);
        assert!((ip.re - 1.0).abs() < 1e-14);
        assert!(ip.im.abs() < 1e-14);
    }

    #[test]
    fn test_basis_state_enum_inner_product_diff_uncoupled() {
        let a = BasisStateEnum::Uncoupled(make_uncoupled(1, 0));
        let b = BasisStateEnum::Uncoupled(make_uncoupled(1, 1));
        let ip = a.inner_product(&b);
        assert!(ip.re.abs() < 1e-14);
    }

    #[test]
    fn test_basis_state_enum_inner_product_same_coupled() {
        let a = BasisStateEnum::Coupled(make_coupled(1, 1, 0));
        let b = BasisStateEnum::Coupled(make_coupled(1, 1, 0));
        let ip = a.inner_product(&b);
        assert!((ip.re - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_transform_to_uncoupled_nonzero() {
        // J=1, I1=1/2(1), I2=1/2(1) -> F1=3/2(3) or 1/2(1)
        // F1=3/2, F can be 1 or 2 (2F=2 or 4)
        let state = CoupledBasisState {
            j: 1,
            f: 1,
            mf: 0,
            i1: 1,
            i2: 1,
            f1: 3,
            omega: 0,
            parity: Some(1),
            electronic_state: ElectronicState::X,
        };
        let uncoupled = state.transform_to_uncoupled();
        assert!(!uncoupled.terms.is_empty());
        let norm_sq: f64 = uncoupled.terms.iter().map(|(a, _)| a.norm_sqr()).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-10,
            "norm^2 = {norm_sq}, expected 1.0"
        );
    }
}
