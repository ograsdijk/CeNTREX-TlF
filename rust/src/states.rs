use num_complex::Complex64;
use std::collections::HashMap;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Sub;
use crate::wigner::clebsch_gordan;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Uncoupled basis state for X state.
pub struct UncoupledBasisState{
    /// Rotational quantum number J
    pub J: i32,
    /// Projection of J on z-axis
    pub mJ: i32,
    /// Nuclear spin I1 (Tl) * 2
    pub I1: i32, // 2 * I1
    /// Projection of I1 on z-axis * 2
    pub m1: i32, // 2 * m1
    /// Nuclear spin I2 (F) * 2
    pub I2: i32, // 2 * I2
    /// Projection of I2 on z-axis * 2
    pub m2: i32, // 2 * m2
    /// Omega quantum number
    pub Omega: i32,
    /// Parity
    pub parity: i8
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
        let mut map: HashMap<UncoupledBasisState, Amp> = HashMap::new();

        for (amp, ket) in iter {
            if remove_zero_amp && amp == Complex64::new(0.0, 0.0) {
                continue;
            }
            *map.entry(ket).or_insert(Complex64::new(0.0, 0.0)) += amp;
        }

        let terms = map
            .into_iter()
            .filter(|(_, amp)| !remove_zero_amp || amp.re != 0.0 || amp.im != 0.0)
            .map(|(ket, amp)| (amp, ket))
            .collect();

        UncoupledState { terms }
    }

    /// Create an empty state.
    pub fn empty() -> Self {
        UncoupledState { terms: Vec::new() }
    }
}


impl Add for UncoupledState {
    type Output = UncoupledState;

    fn add(self, other: UncoupledState) -> UncoupledState {
        let mut map: HashMap<UncoupledBasisState, Amp> = HashMap::new();

        for (amp, ket) in self.terms.into_iter().chain(other.terms.into_iter()) {
            *map.entry(ket).or_insert(Complex64::new(0.0, 0.0)) += amp;
        }

        let terms = map.into_iter()
            .filter(|(_, amp)| amp.re != 0.0 || amp.im != 0.0)
            .map(|(ket, amp)| (amp, ket))
            .collect();

        UncoupledState { terms }
    }
}

impl Sub for UncoupledState {
    type Output = UncoupledState;

    fn sub(self, other: UncoupledState) -> UncoupledState {
        self + (other * -1.0)
    }
}

impl Mul<f64> for UncoupledState {
    type Output = UncoupledState;

    fn mul(self, rhs: f64) -> UncoupledState {
        let terms = self.terms
            .into_iter()
            .map(|(amp, ket)| (amp * rhs, ket))
            .filter(|(amp, _)| amp.re != 0.0 || amp.im != 0.0)
            .collect();
        UncoupledState { terms }
    }
}

impl Mul<Complex64> for UncoupledState {
    type Output = UncoupledState;

    fn mul(self, rhs: Complex64) -> UncoupledState {
        let terms = self.terms
            .into_iter()
            .map(|(amp, ket)| (amp * rhs, ket))
            .filter(|(amp, _)| amp.re != 0.0 || amp.im != 0.0)
            .collect();
        UncoupledState { terms }
    }
}

impl Div<f64> for UncoupledState {
    type Output = UncoupledState;

    fn div(self, rhs: f64) -> UncoupledState {
        self * (1.0 / rhs)
    }
}

impl Mul<UncoupledState> for f64 {
    type Output = UncoupledState;

    fn mul(self, rhs: UncoupledState) -> UncoupledState {
        rhs * self
    }
}

impl Mul<UncoupledState> for Complex64 {
    type Output = UncoupledState;

    fn mul(self, rhs: UncoupledState) -> UncoupledState {
        rhs * self
    }
}
impl Mul<UncoupledState> for UncoupledState {
    type Output = UncoupledState;

    fn mul(self, rhs: UncoupledState) -> UncoupledState {
        let mut map: HashMap<UncoupledBasisState, Amp> = HashMap::new();

        for (amp1, ket1) in self.terms {
            for (amp2, ket2) in &rhs.terms {
                if ket1 == *ket2 {
                    *map.entry(ket1).or_insert(Complex64::new(0.0, 0.0)) += amp1 * amp2;
                }
            }
        }

        let terms = map.into_iter()
            .filter(|(_, amp)| amp.re != 0.0 || amp.im != 0.0)
            .map(|(ket, amp)| (amp, ket))
            .collect();

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
    pub J: i32,
    /// Total angular momentum F
    pub F: i32,
    /// Projection of F on z-axis
    pub mF: i32,
    /// Nuclear spin I1 (Tl) * 2
    pub I1: i32, // 2*I1
    /// Nuclear spin I2 (F) * 2
    pub I2: i32, // 2*I2
    /// Intermediate angular momentum F1 * 2
    pub F1: i32, // 2*F1
    /// Omega quantum number
    pub Omega: i32,
    /// Parity
    pub P: Option<i8>,
    /// Electronic state
    pub electronic_state: ElectronicState,
}

pub type CoupledTerm = (Amp, CoupledBasisState);

#[derive(Clone, Debug)]
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
        let mut map: HashMap<CoupledBasisState, Amp> = HashMap::new();

        for (amp, ket) in iter {
            *map.entry(ket).or_insert(Complex64::new(0.0, 0.0)) += amp;
        }

        let terms = map
            .into_iter()
            .filter(|(_, amp)| amp.re != 0.0 || amp.im != 0.0)
            .map(|(ket, amp)| (amp, ket))
            .collect();

        CoupledState { terms }
    }

    /// Create a state from a vector of terms.
    pub fn from_vec(terms: Vec<CoupledTerm>) -> Self {
        Self::new(terms)
    }
}

impl Add for CoupledState {
    type Output = CoupledState;

    fn add(self, other: CoupledState) -> CoupledState {
        let mut map: HashMap<CoupledBasisState, Amp> = HashMap::new();

        for (amp, ket) in self.terms.into_iter().chain(other.terms.into_iter()) {
            *map.entry(ket).or_insert(Complex64::new(0.0, 0.0)) += amp;
        }

        let terms = map.into_iter()
            .filter(|(_, amp)| amp.re != 0.0 || amp.im != 0.0)
            .map(|(ket, amp)| (amp, ket))
            .collect();

        CoupledState { terms }
    }
}

impl Sub for CoupledState {
    type Output = CoupledState;

    fn sub(self, other: CoupledState) -> CoupledState {
        self + (other * -1.0)
    }
}

impl Mul<f64> for CoupledState {
    type Output = CoupledState;

    fn mul(self, rhs: f64) -> CoupledState {
        let terms = self.terms
            .into_iter()
            .map(|(amp, ket)| (amp * rhs, ket))
            .filter(|(amp, _)| amp.re != 0.0 || amp.im != 0.0)
            .collect();
        CoupledState { terms }
    }
}

impl Mul<Complex64> for CoupledState {
    type Output = CoupledState;

    fn mul(self, rhs: Complex64) -> CoupledState {
        let terms = self.terms
            .into_iter()
            .map(|(amp, ket)| (amp * rhs, ket))
            .filter(|(amp, _)| amp.re != 0.0 || amp.im != 0.0)
            .collect();
        CoupledState { terms }
    }
}

impl Mul<CoupledState> for f64 {
    type Output = CoupledState;

    fn mul(self, rhs: CoupledState) -> CoupledState {
        rhs * self
    }
}

impl Mul<CoupledState> for Complex64 {
    type Output = CoupledState;

    fn mul(self, rhs: CoupledState) -> CoupledState {
        rhs * self
    }
}

impl Div<f64> for CoupledState {
    type Output = CoupledState;

    fn div(self, rhs: f64) -> CoupledState {
        self * (1.0 / rhs)
    }
}

impl CoupledBasisState {
    /// Transform coupled basis state to uncoupled basis.
    pub fn transform_to_uncoupled(&self) -> UncoupledState {
        let J = self.J;
        let F = self.F;
        let mF = self.mF;
        let I1 = self.I1;
        let I2 = self.I2;
        let F1 = self.F1;

        let mut terms = Vec::new();

        // mF1 ranges from -F1 to F1 in steps of 2
        for mF1 in (-F1..=F1).step_by(2) {
            // mJ ranges from -J to J in steps of 1 (since J is integer)
            for mJ in -J..=J {
                // m1 ranges from -I1 to I1 in steps of 2
                for m1 in (-I1..=I1).step_by(2) {
                    // m2 ranges from -I2 to I2 in steps of 2
                    for m2 in (-I2..=I2).step_by(2) {
                        // Check angular momentum conservation
                        // mJ is single, m1, mF1 are doubled
                        if 2 * mJ + m1 != mF1 { continue; }
                        // mF1, m2 are doubled, mF is single
                        if mF1 + m2 != 2 * mF { continue; }

                        // Convert single integer quantum numbers to doubled for CG calculation
                        let cg1 = clebsch_gordan(2 * J, 2 * mJ, I1, m1, F1, mF1);
                        if cg1 == 0.0 { continue; }

                        let cg2 = clebsch_gordan(F1, mF1, I2, m2, 2 * F, 2 * mF);
                        if cg2 == 0.0 { continue; }

                        let amp = Complex64::new(cg1 * cg2, 0.0);
                        let state = UncoupledBasisState {
                            J, mJ, I1, m1, I2, m2,
                            Omega: self.Omega,
                            parity: self.P.unwrap_or(0)
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
    Uncoupled(UncoupledBasisState)
}

impl BasisStateEnum {
    /// Calculate inner product <self|other>.
    pub fn inner_product(&self, other: &BasisStateEnum) -> Complex64 {
        match (self, other) {
            (BasisStateEnum::Coupled(a), BasisStateEnum::Coupled(b)) => {
                if a == b { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) }
            },
            (BasisStateEnum::Uncoupled(a), BasisStateEnum::Uncoupled(b)) => {
                if a == b { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) }
            },
            (BasisStateEnum::Uncoupled(a), BasisStateEnum::Coupled(b)) => {
                let uncoupled_b = b.transform_to_uncoupled();
                for (amp, state) in uncoupled_b.terms {
                    if state == *a {
                        return amp;
                    }
                }
                Complex64::new(0.0, 0.0)
            },
            (BasisStateEnum::Coupled(a), BasisStateEnum::Uncoupled(b)) => {
                let uncoupled_a = a.transform_to_uncoupled();
                for (amp, state) in uncoupled_a.terms {
                    if state == *b {
                        return amp.conj();
                    }
                }
                Complex64::new(0.0, 0.0)
            }
        }
    }
}

