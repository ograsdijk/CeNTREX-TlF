use crate::b_coupled;
use crate::constants::{BConstants, XConstants};
use crate::states::{CoupledBasisState, CoupledState, UncoupledBasisState, UncoupledState};
use crate::x_uncoupled;
use num_complex::Complex64;
use rayon::prelude::*;
use std::collections::HashMap;
use std::hash::Hash;

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
    type BasisState: Copy + Eq + Hash;
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
    B: Copy + Eq + Hash,
    S: OperatorState<BasisState = B>,
{
    let h_applied: Vec<S> = qn.iter().map(|b| h(*b, constants)).collect();
    h_mat_elems_from_applied(qn, &h_applied)
}

fn h_mat_elems_from_applied<B, S>(qn: &[B], h_applied: &[S]) -> Vec<Complex64>
where
    B: Copy + Eq + Hash,
    S: OperatorState<BasisState = B>,
{
    let n = qn.len();
    debug_assert_eq!(n, h_applied.len());
    let mut result = vec![Complex64::ZERO; n * n];
    // One index over the basis, then scatter each applied state's few terms:
    // O(n * terms) instead of n^2 / 2 hashed lookups. Amplitudes accumulate in
    // the same term order as before, so the upper triangle is bit-identical;
    // the lower triangle is still its conjugate.
    let index: HashMap<B, usize> = qn.iter().enumerate().map(|(i, &b)| (b, i)).collect();
    if index.len() != n {
        return h_mat_elems_from_applied_per_column(qn, h_applied);
    }
    for (j, psi) in h_applied.iter().enumerate() {
        for &(amp, basis) in psi.terms() {
            if let Some(&i) = index.get(&basis) {
                if i <= j {
                    result[i * n + j] += amp;
                }
            }
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            result[j * n + i] = result[i * n + j].conj();
        }
    }
    result
}

/// Per-column lookup maps; only used when `qn` repeats a basis state, where a
/// single basis -> index map cannot represent every row.
fn h_mat_elems_from_applied_per_column<B, S>(qn: &[B], h_applied: &[S]) -> Vec<Complex64>
where
    B: Copy + Eq + Hash,
    S: OperatorState<BasisState = B>,
{
    let n = qn.len();
    let mut result = vec![Complex64::ZERO; n * n];
    let lookups: Vec<HashMap<B, Complex64>> = h_applied
        .iter()
        .map(|psi| {
            let mut lookup = HashMap::with_capacity(psi.terms().len());
            for &(amp, basis) in psi.terms() {
                *lookup.entry(basis).or_insert(Complex64::ZERO) += amp;
            }
            lookup
        })
        .collect();

    for (i, a) in qn.iter().enumerate() {
        for j in i..n {
            let val = lookups[j].get(a).copied().unwrap_or(Complex64::ZERO);
            result[i * n + j] = val;
            if i != j {
                result[j * n + i] = val.conj();
            }
        }
    }
    result
}

/// Linear-scan reference for `h_mat_elems_from_applied`, kept for benchmarking.
///
/// Instead of indexing the basis once and scattering each applied state's
/// terms, it rescans the term list for every matrix element. Retained under `cfg(test)` so the equivalence test and the
/// `bench_h_mat_elems_lookup_vs_linear_scan` measurement have something to
/// compare against; it is not part of the shipped path.
#[cfg(test)]
fn h_mat_elems_from_applied_linear_scan<B, S>(qn: &[B], h_applied: &[S]) -> Vec<Complex64>
where
    B: Copy + Eq + Hash,
    S: OperatorState<BasisState = B>,
{
    let n = qn.len();
    debug_assert_eq!(n, h_applied.len());
    let mut result = vec![Complex64::ZERO; n * n];
    for (i, a) in qn.iter().enumerate() {
        for j in i..n {
            let mut val = Complex64::ZERO;
            for &(amp, basis) in h_applied[j].terms() {
                if basis == *a {
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
    ];

    let results: Vec<Vec<Complex64>> = ops
        .into_par_iter()
        .map(|op| h_mat_elems_b(op, qn, constants))
        .collect();

    let component_results = rayon::join(
        || {
            let components: Vec<(CoupledState, CoupledState, CoupledState)> = qn
                .par_iter()
                .map(|basis| b_coupled::stark_components(*basis, constants))
                .collect();
            let mut sx = Vec::with_capacity(components.len());
            let mut sy = Vec::with_capacity(components.len());
            let mut sz = Vec::with_capacity(components.len());
            for (x, y, z) in components {
                sx.push(x);
                sy.push(y);
                sz.push(z);
            }
            (
                h_mat_elems_from_applied(qn, &sx),
                h_mat_elems_from_applied(qn, &sy),
                h_mat_elems_from_applied(qn, &sz),
            )
        },
        || {
            let components: Vec<(CoupledState, CoupledState, CoupledState)> = qn
                .par_iter()
                .map(|basis| b_coupled::zeeman_components(*basis, constants))
                .collect();
            let mut zx = Vec::with_capacity(components.len());
            let mut zy = Vec::with_capacity(components.len());
            let mut zz = Vec::with_capacity(components.len());
            for (x, y, z) in components {
                zx.push(x);
                zy.push(y);
                zz.push(z);
            }
            (
                h_mat_elems_from_applied(qn, &zx),
                h_mat_elems_from_applied(qn, &zy),
                h_mat_elems_from_applied(qn, &zz),
            )
        },
    );

    let mut it = results.into_iter();
    let (stark, zeeman) = component_results;
    HamiltonianCoupledB {
        h_rot: it.next().unwrap(),
        h_mhf_tl: it.next().unwrap(),
        h_mhf_f: it.next().unwrap(),
        h_ld: it.next().unwrap(),
        h_cp1_tl: it.next().unwrap(),
        h_c_tl: it.next().unwrap(),
        h_sx: stark.0,
        h_sy: stark.1,
        h_sz: stark.2,
        h_zx: zeeman.0,
        h_zy: zeeman.1,
        h_zz: zeeman.2,
    }
}

/// Signed-Omega expansion of a set of parity-basis B states.
///
/// `|J, |Omega|, P> = (|J, +Omega> + P (-1)^J |J, -Omega>) / sqrt(2)`, so every
/// parity state has exactly two Omega-basis components of magnitude 1/sqrt(2).
/// The common magnitude is left out here and applied once in the transform.
struct ParityExpansion {
    /// Omega-basis states, in order of first appearance
    omega_basis: Vec<CoupledBasisState>,
    /// Index into `omega_basis` of the +Omega component of each parity state
    plus: Vec<usize>,
    /// Index into `omega_basis` of the -Omega component of each parity state
    minus: Vec<usize>,
    /// Sign `P (-1)^J` of the -Omega component (the +Omega sign is +1)
    minus_sign: Vec<f64>,
}

fn expand_parity_basis(qn: &[CoupledBasisState]) -> Result<ParityExpansion, String> {
    let mut omega_basis: Vec<CoupledBasisState> = Vec::with_capacity(2 * qn.len());
    let mut index: HashMap<CoupledBasisState, usize> = HashMap::with_capacity(2 * qn.len());
    let mut plus = Vec::with_capacity(qn.len());
    let mut minus = Vec::with_capacity(qn.len());
    let mut minus_sign = Vec::with_capacity(qn.len());

    let mut index_of = |state: CoupledBasisState, omega_basis: &mut Vec<CoupledBasisState>| {
        *index.entry(state).or_insert_with(|| {
            omega_basis.push(state);
            omega_basis.len() - 1
        })
    };

    for state in qn {
        let parity = state
            .parity
            .ok_or_else(|| "parity-basis states need a parity (P)".to_string())?;
        if parity != 1 && parity != -1 {
            return Err(format!("parity must be +1 or -1, got {parity}"));
        }
        if state.omega == 0 || state.electronic_state != crate::states::ElectronicState::B {
            return Err(
                "only B states with nonzero Omega have a parity-basis expansion".to_string(),
            );
        }
        let mut state_plus = *state;
        state_plus.parity = None;
        let mut state_minus = state_plus;
        state_minus.omega = -state.omega;

        plus.push(index_of(state_plus, &mut omega_basis));
        minus.push(index_of(state_minus, &mut omega_basis));
        let j_sign = if state.j % 2 == 0 { 1.0 } else { -1.0 };
        minus_sign.push(parity as f64 * j_sign);
    }

    Ok(ParityExpansion {
        omega_basis,
        plus,
        minus,
        minus_sign,
    })
}

/// `T^dagger M T` for the parity expansion, with `M` an `n_omega x n_omega`
/// row-major matrix, followed by the same Hermitian symmetrization the Python
/// path applies. Each element is a signed sum of four elements of `M`.
fn omega_matrix_to_parity(m: &[Complex64], expansion: &ParityExpansion) -> Vec<Complex64> {
    let n_omega = expansion.omega_basis.len();
    let n = expansion.plus.len();
    debug_assert_eq!(m.len(), n_omega * n_omega);

    let mut transformed = vec![Complex64::ZERO; n * n];
    for i in 0..n {
        let row_plus = expansion.plus[i] * n_omega;
        let row_minus = expansion.minus[i] * n_omega;
        let sign_i = expansion.minus_sign[i];
        for j in 0..n {
            let col_plus = expansion.plus[j];
            let col_minus = expansion.minus[j];
            // (row combination) first, then (column combination): the order the
            // sparse products in Python accumulate in.
            let left = m[row_plus + col_plus] + m[row_minus + col_plus] * sign_i;
            let right = m[row_plus + col_minus] + m[row_minus + col_minus] * sign_i;
            transformed[i * n + j] = left + right * expansion.minus_sign[j];
        }
    }

    // amplitude 1/sqrt(2) on both sides, written as Python evaluates it
    let amplitude = 1.0 / 2.0_f64.sqrt();
    let factor = 0.5 * (amplitude * amplitude);
    let mut result = vec![Complex64::ZERO; n * n];
    for i in 0..n {
        for j in 0..n {
            result[i * n + j] = (transformed[i * n + j] + transformed[j * n + i].conj()) * factor;
        }
    }
    result
}

/// B Hamiltonian for parity-basis states: built in the signed-Omega basis the
/// operators are written in, then transformed.
pub fn generate_coupled_hamiltonian_b_parity(
    qn: &[CoupledBasisState],
    constants: &BConstants,
) -> Result<HamiltonianCoupledB, String> {
    let expansion = expand_parity_basis(qn)?;
    let omega = generate_coupled_hamiltonian_b(&expansion.omega_basis, constants);

    let matrices = [
        &omega.h_rot,
        &omega.h_mhf_tl,
        &omega.h_mhf_f,
        &omega.h_ld,
        &omega.h_cp1_tl,
        &omega.h_c_tl,
        &omega.h_sx,
        &omega.h_sy,
        &omega.h_sz,
        &omega.h_zx,
        &omega.h_zy,
        &omega.h_zz,
    ];
    let results: Vec<Vec<Complex64>> = matrices
        .into_par_iter()
        .map(|m| omega_matrix_to_parity(m, &expansion))
        .collect();

    let mut it = results.into_iter();
    Ok(HamiltonianCoupledB {
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
    })
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

    /// Physically correct X uncoupled basis for J = 0..=jmax.
    ///
    /// `make_x_basis` above steps mj by 2, which drops half the mJ values --
    /// harmless for a Hermiticity check, but it would misstate the basis size
    /// the lookup benchmark is trying to characterize. Sizes here are the real
    /// ones: 4*(jmax+1)^2, i.e. 4, 16, 36, 64, 100, 144, 196, 256.
    fn make_full_x_basis(jmax: i32) -> Vec<UncoupledBasisState> {
        let mut states = Vec::new();
        for j in 0..=jmax {
            for mj in -j..=j {
                for m1 in [-1, 1] {
                    for m2 in [-1, 1] {
                        states.push(UncoupledBasisState {
                            j,
                            mj,
                            i1: 1,
                            m1,
                            i2: 1,
                            m2,
                            omega: 0,
                            parity: 1,
                        });
                    }
                }
            }
        }
        states
    }

    /// B coupled basis for J = 1..=jmax, Omega basis (parity None, omega = 1).
    ///
    /// F1 = J +- 1/2 and F = F1 +- 1/2, both stored doubled for F1 and single
    /// for F, matching `CoupledBasisState`.
    fn make_b_basis(jmax: i32) -> Vec<CoupledBasisState> {
        let mut states = Vec::new();
        for j in 1..=jmax {
            for f1 in [2 * j - 1, 2 * j + 1] {
                if f1 <= 0 {
                    continue;
                }
                for f in [(f1 - 1) / 2, (f1 + 1) / 2] {
                    for mf in -f..=f {
                        states.push(CoupledBasisState {
                            j,
                            f,
                            mf,
                            i1: 1,
                            i2: 1,
                            f1,
                            omega: 1,
                            parity: None,
                            electronic_state: ElectronicState::B,
                        });
                    }
                }
            }
        }
        states
    }

    #[test]
    fn test_linear_scan_reference_matches_lookup_maps() {
        // Pins the benchmark's baseline: if these ever disagree, the timing
        // comparison below is meaningless.
        let constants = XConstants::default();
        let ops: [fn(UncoupledBasisState, &XConstants) -> UncoupledState; 4] = [
            x_uncoupled::h_ff,
            x_uncoupled::h_sx,
            x_uncoupled::h_sy,
            x_uncoupled::h_zz,
        ];
        for jmax in [0, 1, 2, 3] {
            let qn = make_full_x_basis(jmax);
            for op in ops {
                let applied: Vec<UncoupledState> = qn.iter().map(|b| op(*b, &constants)).collect();
                let mapped = h_mat_elems_from_applied(&qn, &applied);
                let scanned = h_mat_elems_from_applied_linear_scan(&qn, &applied);
                assert_eq!(mapped.len(), scanned.len());
                for (idx, (a, b)) in mapped.iter().zip(scanned.iter()).enumerate() {
                    assert!(
                        (a - b).norm() < 1e-12,
                        "jmax={jmax} idx={idx}: {a:?} vs {b:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_index_scatter_is_bit_identical_to_per_column_maps() {
        let constants = XConstants::default();
        let ops: [fn(UncoupledBasisState, &XConstants) -> UncoupledState; 4] = [
            x_uncoupled::h_ff,
            x_uncoupled::h_sx,
            x_uncoupled::h_sy,
            x_uncoupled::h_zz,
        ];
        for jmax in [0, 1, 2, 3, 4] {
            let qn = make_full_x_basis(jmax);
            for op in ops {
                let applied: Vec<UncoupledState> = qn.iter().map(|b| op(*b, &constants)).collect();
                assert_eq!(
                    h_mat_elems_from_applied(&qn, &applied),
                    h_mat_elems_from_applied_per_column(&qn, &applied),
                    "jmax={jmax}"
                );
            }
        }
    }

    /// Per-trial timings for both implementations, interleaved.
    ///
    /// Map and scan alternate *within* each trial rather than running as two
    /// separate blocks, so slow drift (thermal throttling, a background task)
    /// hits both roughly equally instead of loading onto whichever ran second.
    /// Returns (map_us_per_call, scan_us_per_call) with one entry per trial.
    #[cfg(test)]
    fn time_map_vs_scan<B, S>(
        qn: &[B],
        applied: &[S],
        reps: usize,
        trials: usize,
    ) -> (Vec<f64>, Vec<f64>)
    where
        B: Copy + Eq + std::hash::Hash,
        S: OperatorState<BasisState = B>,
    {
        use std::hint::black_box;
        use std::time::Instant;

        let mut map_us = Vec::with_capacity(trials);
        let mut scan_us = Vec::with_capacity(trials);
        for _ in 0..trials {
            // black_box on both the inputs and the whole result matrix:
            // reading only m[0] would let LLVM strip the linear scan down to a
            // single element while the opaque HashMap survives, which would
            // fake a win for the scan.
            let start = Instant::now();
            for _ in 0..reps {
                let m = h_mat_elems_from_applied(black_box(qn), black_box(applied));
                black_box(&m);
            }
            map_us.push(start.elapsed().as_secs_f64() * 1e6 / reps as f64);

            let start = Instant::now();
            for _ in 0..reps {
                let m = h_mat_elems_from_applied_linear_scan(black_box(qn), black_box(applied));
                black_box(&m);
            }
            scan_us.push(start.elapsed().as_secs_f64() * 1e6 / reps as f64);
        }
        (map_us, scan_us)
    }

    #[cfg(test)]
    fn median(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            0.5 * (sorted[mid - 1] + sorted[mid])
        } else {
            sorted[mid]
        }
    }

    /// Spread as a fraction of the median, i.e. (max - min) / median.
    #[cfg(test)]
    fn rel_spread(values: &[f64]) -> f64 {
        let lo = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (hi - lo) / median(values)
    }

    /// Measurement for the audit item "benchmark the h_mat_elems_generic
    /// lookup maps against the old linear scan for very small bases".
    ///
    /// Ignored by default -- it is a timing report, not a pass/fail assertion.
    /// Run with:
    ///     cargo test --release -p centrex_tlf_rust h_mat_elems_lookup -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_h_mat_elems_lookup_vs_linear_scan() {
        const TRIALS: usize = 7;

        let constants = XConstants::default();
        let ops: [(&str, fn(UncoupledBasisState, &XConstants) -> UncoupledState); 3] = [
            ("h_ff", x_uncoupled::h_ff),
            ("h_sx", x_uncoupled::h_sx),
            ("h_zz", x_uncoupled::h_zz),
        ];

        // The comparison hinges on terms-per-applied-state k: the map pays
        // n hash-map builds plus n^2/2 hashed lookups, the scan pays
        // n^2/2 * k struct compares. X operators have small k, so B (larger
        // k from the F1'/F' sums) is measured too rather than assumed.
        //
        // Columns: median over TRIALS interleaved trials, each averaging over
        // `reps` calls; "sprd" is (max - min) / median across trials, so it
        // says how much the medians can be trusted. "ratio range" is the
        // min..max of the per-trial scan/map ratio -- the honest bound on the
        // headline number.
        println!("trials = {TRIALS}, interleaved; times are us per call");
        println!(
            "{:>3} {:>5} {:>6} {:>10} {:>7} {:>11} {:>6} {:>11} {:>6} {:>8} {:>13}",
            "sp",
            "jmax",
            "n",
            "op",
            "terms",
            "map med",
            "sprd",
            "scan med",
            "sprd",
            "scan/map",
            "ratio range"
        );

        let mut worst_ratio: f64 = 0.0;
        let mut worst_spread: f64 = 0.0;

        for jmax in [0, 1, 2, 3, 4, 5, 6, 7] {
            let qn = make_full_x_basis(jmax);
            let n = qn.len();
            for (name, op) in ops {
                let applied: Vec<UncoupledState> = qn.iter().map(|b| op(*b, &constants)).collect();
                let terms: usize = applied.iter().map(|s| s.terms().len()).sum();
                // Repeat enough that the smallest bases are not pure timer noise.
                let reps = (2_000_000 / (n * n).max(1)).clamp(5, 20_000);
                let (map_us, scan_us) = time_map_vs_scan(&qn, &applied, reps, TRIALS);
                let ratios: Vec<f64> = map_us
                    .iter()
                    .zip(scan_us.iter())
                    .map(|(m, s)| s / m)
                    .collect();
                let ratio_lo = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
                let ratio_hi = ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                worst_ratio = worst_ratio.max(ratio_hi);
                worst_spread = worst_spread
                    .max(rel_spread(&map_us))
                    .max(rel_spread(&scan_us));
                println!(
                    "{:>3} {:>5} {:>6} {:>10} {:>7.2} {:>11.3} {:>5.1}% {:>11.3} {:>5.1}% {:>8.2} {:>6.2}..{:<5.2}",
                    "X",
                    jmax,
                    n,
                    name,
                    terms as f64 / n as f64,
                    median(&map_us),
                    100.0 * rel_spread(&map_us),
                    median(&scan_us),
                    100.0 * rel_spread(&scan_us),
                    median(&scan_us) / median(&map_us),
                    ratio_lo,
                    ratio_hi
                );
            }
        }

        let b_constants = BConstants::default();
        let b_ops: [(&str, fn(CoupledBasisState, &BConstants) -> CoupledState); 4] = [
            ("h_mhf_tl", b_coupled::h_mhf_tl),
            ("h_mhf_f", b_coupled::h_mhf_f),
            ("h_c_tl", b_coupled::h_c_tl),
            ("h_zz", b_coupled::h_zz),
        ];
        for jmax in [1, 2, 3, 4, 6, 8] {
            let qn = make_b_basis(jmax);
            let n = qn.len();
            for (name, op) in b_ops {
                let applied: Vec<CoupledState> = qn.iter().map(|b| op(*b, &b_constants)).collect();
                let terms: usize = applied.iter().map(|s| s.terms().len()).sum();
                let reps = (2_000_000 / (n * n).max(1)).clamp(5, 20_000);
                let (map_us, scan_us) = time_map_vs_scan(&qn, &applied, reps, TRIALS);
                let ratios: Vec<f64> = map_us
                    .iter()
                    .zip(scan_us.iter())
                    .map(|(m, s)| s / m)
                    .collect();
                let ratio_lo = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
                let ratio_hi = ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                worst_ratio = worst_ratio.max(ratio_hi);
                worst_spread = worst_spread
                    .max(rel_spread(&map_us))
                    .max(rel_spread(&scan_us));
                println!(
                    "{:>3} {:>5} {:>6} {:>10} {:>7.2} {:>11.3} {:>5.1}% {:>11.3} {:>5.1}% {:>8.2} {:>6.2}..{:<5.2}",
                    "B",
                    jmax,
                    n,
                    name,
                    terms as f64 / n as f64,
                    median(&map_us),
                    100.0 * rel_spread(&map_us),
                    median(&scan_us),
                    100.0 * rel_spread(&scan_us),
                    median(&scan_us) / median(&map_us),
                    ratio_lo,
                    ratio_hi
                );
            }
        }

        println!();
        println!(
            "worst single-trial scan/map ratio: {worst_ratio:.3}               (>= 1.0 would mean the map won a trial)"
        );
        println!(
            "worst per-cell relative spread across trials: {:.1}%",
            100.0 * worst_spread
        );
        // The per-column HashMaps this benchmark was written against lost to
        // the scan everywhere. The shipped path is now a single basis index
        // plus a scatter; it is within noise of the scan below n ~ 100 and up
        // to ~2x faster for X at n = 144-196. Every case is well under a
        // millisecond, so this is a report, not a pass/fail gate.
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

    /// Parity basis for J = 1..=jmax, both parities, Omega = 1.
    fn make_b_parity_basis(jmax: i32) -> Vec<CoupledBasisState> {
        let mut states = Vec::new();
        for parity in [-1i8, 1] {
            for mut state in make_b_basis(jmax) {
                state.parity = Some(parity);
                states.push(state);
            }
        }
        states
    }

    #[test]
    fn test_parity_hamiltonian_matches_explicit_transform() {
        let constants = BConstants::default();
        let qn = make_b_parity_basis(3);
        let n = qn.len();
        let expansion = expand_parity_basis(&qn).unwrap();
        let n_omega = expansion.omega_basis.len();
        assert_eq!(n_omega, n);

        let omega = generate_coupled_hamiltonian_b(&expansion.omega_basis, &constants);
        let parity = generate_coupled_hamiltonian_b_parity(&qn, &constants).unwrap();

        // Dense T (n_omega x n), then T^dagger M T the slow way.
        let amplitude = 1.0 / 2.0_f64.sqrt();
        let mut t = vec![0.0; n_omega * n];
        for i in 0..n {
            t[expansion.plus[i] * n + i] += amplitude;
            t[expansion.minus[i] * n + i] += amplitude * expansion.minus_sign[i];
        }
        for (m, got) in [
            (&omega.h_rot, &parity.h_rot),
            (&omega.h_ld, &parity.h_ld),
            (&omega.h_mhf_f, &parity.h_mhf_f),
            (&omega.h_sz, &parity.h_sz),
            (&omega.h_sy, &parity.h_sy),
            (&omega.h_zx, &parity.h_zx),
        ] {
            let scale = m.iter().map(|v| v.norm()).fold(0.0, f64::max);
            for i in 0..n {
                for j in 0..n {
                    let mut expected = Complex64::ZERO;
                    for k in 0..n_omega {
                        if t[k * n + i] == 0.0 {
                            continue;
                        }
                        for l in 0..n_omega {
                            expected += m[k * n_omega + l] * (t[k * n + i] * t[l * n + j]);
                        }
                    }
                    let diff = (expected - got[i * n + j]).norm();
                    assert!(
                        diff <= 1e-12 * scale,
                        "({i},{j}): {expected:?} vs {:?}",
                        got[i * n + j]
                    );
                    let asym = (got[i * n + j] - got[j * n + i].conj()).norm();
                    assert!(asym == 0.0, "not Hermitian at ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn test_parity_hamiltonian_is_block_diagonal_in_parity_without_field() {
        // Field-free terms conserve parity; Stark terms connect opposite parity only.
        let constants = BConstants::default();
        let qn = make_b_parity_basis(2);
        let n = qn.len();
        let ham = generate_coupled_hamiltonian_b_parity(&qn, &constants).unwrap();
        for i in 0..n {
            for j in 0..n {
                let same_parity = qn[i].parity == qn[j].parity;
                if same_parity {
                    assert_eq!(ham.h_sz[i * n + j], Complex64::ZERO);
                } else {
                    assert_eq!(ham.h_rot[i * n + j], Complex64::ZERO);
                    assert_eq!(ham.h_ld[i * n + j], Complex64::ZERO);
                    assert_eq!(ham.h_mhf_tl[i * n + j], Complex64::ZERO);
                }
            }
        }
    }

    #[test]
    fn test_parity_hamiltonian_rejects_states_without_parity() {
        let constants = BConstants::default();
        let qn = make_b_basis(1);
        assert!(generate_coupled_hamiltonian_b_parity(&qn, &constants).is_err());
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
