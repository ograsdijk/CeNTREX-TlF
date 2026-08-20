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
/// This is what the map-based version replaced: instead of hashing each applied
/// state's terms into a `HashMap` once, it rescans the term list for every
/// matrix element. Retained under `cfg(test)` so the equivalence test and the
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
        // The conclusion is "scan is faster everywhere", so the meaningful
        // check is that no individual trial ever went the other way -- not
        // that the medians happen to be separated.
        assert!(
            worst_ratio < 1.0,
            "a trial had the map at least as fast (ratio {worst_ratio:.3});              the report's conclusion needs revisiting"
        );
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
