use crate::lindblad::solver_stats::SolveStats;
use num_complex::Complex64;

#[derive(Clone, Debug)]
pub enum FastOutputKind {
    Full,
    Populations,
    Selected(Vec<(usize, usize)>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastOutputWhen {
    Saveat,
    Final,
}

#[derive(Clone, Debug)]
pub struct FastOutputOptions {
    pub kind: FastOutputKind,
    pub when: FastOutputWhen,
    pub dense_output: bool,
}

impl Default for FastOutputOptions {
    fn default() -> Self {
        Self {
            kind: FastOutputKind::Full,
            when: FastOutputWhen::Saveat,
            dense_output: true,
        }
    }
}

#[derive(Clone, Debug)]
pub enum FastOutputValues {
    Full(Vec<f64>),
    Real(Vec<f64>),
    Complex(Vec<Complex64>),
}

#[derive(Clone, Debug)]
pub struct FastSolveOutput {
    pub times: Vec<f64>,
    pub values: FastOutputValues,
    pub width: usize,
    pub stats: SolveStats,
}
