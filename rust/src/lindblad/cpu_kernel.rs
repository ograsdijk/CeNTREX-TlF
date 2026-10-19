//! Runtime-selected packed D+S+U kernels. ISA validation happens during plan
//! preparation, never inside the ODE stage loop. No global target CPU flags.
use crate::lindblad::layout::UpperTriLayout;
use crate::lindblad::plan::PreparedLindbladPlan;
use num_complex::Complex64;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum KernelKind {
    Original,
    Avx2,
    Avx2Fma,
    Avx512,
    Avx512Fma,
}

/// A validated ISA choice. The private field prevents safe callers from
/// constructing an unsupported kernel and reaching target-feature functions.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CpuKernel {
    kind: KernelKind,
}

#[derive(Copy, Clone, Default)]
struct Features {
    avx2: bool,
    fma: bool,
    avx512f: bool,
}

impl Features {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::is_x86_feature_detected!("avx2"),
                fma: std::is_x86_feature_detected!("fma"),
                avx512f: std::is_x86_feature_detected!("avx512f"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self::default()
        }
    }
}

fn choose(request: &str, f: Features) -> Result<KernelKind, String> {
    let kind=match request {
        "auto" => if f.avx2 && f.fma {KernelKind::Avx2Fma}
                  else if f.avx2 {KernelKind::Avx2} else {KernelKind::Original},
        "original" => KernelKind::Original,
        "avx2" if f.avx2 => KernelKind::Avx2,
        "avx2_fma" if f.avx2 && f.fma => KernelKind::Avx2Fma,
        "avx512" if f.avx2 && f.fma && f.avx512f => KernelKind::Avx512,
        "avx512_fma" if f.avx2 && f.fma && f.avx512f => KernelKind::Avx512Fma,
        "avx2" | "avx2_fma" | "avx512" | "avx512_fma" =>
            return Err(format!("cpu_kernel={request:?} is not supported by this CPU/OS (AVX2={}, FMA={}, AVX512F={})",f.avx2,f.fma,f.avx512f)),
        _=>return Err(format!("unsupported cpu_kernel {request:?}; expected auto, original, avx2, avx2_fma, avx512, or avx512_fma")),
    };
    Ok(kind)
}

impl CpuKernel {
    pub fn resolve(request: &str) -> Result<Self, String> {
        choose(request, Features::detect()).map(|kind| Self { kind })
    }
    pub fn is_original(self) -> bool {
        self.kind == KernelKind::Original
    }
    pub fn as_str(self) -> &'static str {
        match self.kind {
            KernelKind::Original => "original",
            KernelKind::Avx2 => "avx2",
            KernelKind::Avx2Fma => "avx2_fma",
            KernelKind::Avx512 => "avx512",
            KernelKind::Avx512Fma => "avx512_fma",
        }
    }
}

#[inline]
fn row_update(kernel: CpuKernel, h: Complex64, rho: &[Complex64], re: &mut [f64], im: &mut [f64]) {
    debug_assert_eq!(rho.len(), re.len());
    debug_assert_eq!(rho.len(), im.len());
    #[cfg(target_arch = "x86_64")]
    // SAFETY: CpuKernel can only be created by the feature-checked resolver.
    // All row slices have equal lengths; helpers bound each load by rho.len().
    unsafe {
        match kernel.kind {
            KernelKind::Avx2 => {
                avx2_row(h, rho, re, im);
                return;
            }
            KernelKind::Avx2Fma => {
                avx2_fma_row(h, rho, re, im);
                return;
            }
            KernelKind::Avx512 => {
                avx512_row(h, rho, re, im, false);
                return;
            }
            KernelKind::Avx512Fma => {
                avx512_row(h, rho, re, im, true);
                return;
            }
            KernelKind::Original => {}
        }
    }
    scalar_row(h, rho, re, im);
}

/// Sparse off-diagonal H*rho, followed by vectorized diagonal evolution,
/// structured decay and stores directly into the solver's packed output.
#[allow(clippy::too_many_arguments)]
pub(super) fn packed_rhs(
    plan: &PreparedLindbladPlan,
    upper: &UpperTriLayout,
    h_upper: &[Complex64],
    h_sparse: &[Complex64],
    rho: &[Complex64],
    scratch_re: &mut [f64],
    scratch_im: &mut [f64],
    diagonal: &mut [f64],
    out: &mut [f64],
) -> Result<(), String> {
    let n = plan.layout.n;
    let size = n * n;
    // Validate every buffer used by unsafe vector loads, including callers
    // constructing a plan directly in Rust rather than through the parser.
    if upper.n != n
        || h_upper.len() != upper.len()
        || rho.len() != size
        || scratch_re.len() != size
        || scratch_im.len() != size
        || diagonal.len() != n
        || out.len() != size
        || plan.source_decay_rates.len() != n
        || plan.incoming_transfers_by_target.len() != n
    {
        return Err("invalid buffer dimensions for packed CPU kernel".into());
    }
    if plan.cpu_kernel.is_original() {
        return Err("original CPU kernel must use the original RHS path".into());
    }
    let p = &plan.hamiltonian_sparse_pattern;
    if p.n != n || h_sparse.len() != p.nnz {
        return Err("invalid Hamiltonian sparsity dimensions for CPU kernel".into());
    }
    scratch_re.fill(0.0);
    scratch_im.fill(0.0);
    for i in 0..n {
        for ptr in p.row_ptrs[i]..p.row_ptrs[i + 1] {
            let k = p.col_indices[ptr];
            let h = h_sparse[ptr];
            if k == i || h == Complex64::ZERO {
                continue;
            }
            row_update(
                plan.cpu_kernel,
                h,
                &rho[k * n..(k + 1) * n],
                &mut scratch_re[i * n..(i + 1) * n],
                &mut scratch_im[i * n..(i + 1) * n],
            );
        }
        for cp in p.col_ptrs[i]..p.col_ptrs[i + 1] {
            let k = p.row_indices_csc[cp];
            let h = h_sparse[p.csc_to_csr[cp]].conj();
            if h == Complex64::ZERO {
                continue;
            }
            row_update(
                plan.cpu_kernel,
                h,
                &rho[k * n..(k + 1) * n],
                &mut scratch_re[i * n..(i + 1) * n],
                &mut scratch_im[i * n..(i + 1) * n],
            );
        }
    }
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: all optimized choices require AVX2, checked once by resolve.
        // Buffer lengths were checked above; j+4<=n bounds all contiguous loads
        // and transposed gathers. Scalar transfer indexing is bounds-checked.
        unsafe {
            finalize_packed(
                plan, upper, h_upper, rho, scratch_re, scratch_im, diagonal, out,
            );
        }
        Ok(())
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Err("optimized CPU kernels require x86_64".into())
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    #[test]
    fn auto_is_conservative_and_portable() {
        assert_eq!(
            choose("auto", Features::default()).unwrap(),
            KernelKind::Original
        );
        assert_eq!(
            choose(
                "auto",
                Features {
                    avx2: true,
                    ..Features::default()
                }
            )
            .unwrap(),
            KernelKind::Avx2
        );
        assert_eq!(
            choose(
                "auto",
                Features {
                    avx2: true,
                    fma: true,
                    avx512f: true
                }
            )
            .unwrap(),
            KernelKind::Avx2Fma
        );
    }
    #[test]
    fn explicit_features_are_required() {
        for name in ["avx2", "avx2_fma", "avx512", "avx512_fma"] {
            assert!(choose(name, Features::default()).is_err());
        }
        assert!(choose(
            "avx2_fma",
            Features {
                avx2: true,
                ..Features::default()
            }
        )
        .is_err());
        assert!(choose(
            "avx512",
            Features {
                avx2: true,
                fma: true,
                ..Features::default()
            }
        )
        .is_err());
        assert_eq!(
            choose(
                "avx512_fma",
                Features {
                    avx2: true,
                    fma: true,
                    avx512f: true
                }
            )
            .unwrap(),
            KernelKind::Avx512Fma
        );
        assert!(choose("fastest", Features::default()).is_err());
        assert_eq!(
            choose("original", Features::default()).unwrap(),
            KernelKind::Original
        );
    }
}

#[inline]
fn scalar_row(h: Complex64, rho: &[Complex64], re: &mut [f64], im: &mut [f64]) {
    for j in 0..rho.len() {
        let r = rho[j];
        re[j] += h.re * r.re - h.im * r.im;
        im[j] += h.re * r.im + h.im * r.re;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_row(h: Complex64, rho: &[Complex64], re: &mut [f64], im: &mut [f64]) {
    use std::arch::x86_64::*;
    let hr = _mm256_set1_pd(h.re);
    let hi = _mm256_set1_pd(h.im);
    let mut j = 0;
    while j + 4 <= rho.len() {
        let a = _mm256_permute4x64_pd(_mm256_loadu_pd(rho.as_ptr().add(j).cast()), 0xd8);
        let b = _mm256_permute4x64_pd(_mm256_loadu_pd(rho.as_ptr().add(j + 2).cast()), 0xd8);
        let rr = _mm256_permute2f128_pd(a, b, 0x20);
        let ri = _mm256_permute2f128_pd(a, b, 0x31);
        let dr = _mm256_sub_pd(_mm256_mul_pd(hr, rr), _mm256_mul_pd(hi, ri));
        let di = _mm256_add_pd(_mm256_mul_pd(hr, ri), _mm256_mul_pd(hi, rr));
        _mm256_storeu_pd(
            re.as_mut_ptr().add(j),
            _mm256_add_pd(_mm256_loadu_pd(re.as_ptr().add(j)), dr),
        );
        _mm256_storeu_pd(
            im.as_mut_ptr().add(j),
            _mm256_add_pd(_mm256_loadu_pd(im.as_ptr().add(j)), di),
        );
        j += 4;
    }
    scalar_row(h, &rho[j..], &mut re[j..], &mut im[j..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_fma_row(h: Complex64, rho: &[Complex64], re: &mut [f64], im: &mut [f64]) {
    use std::arch::x86_64::*;
    let hr = _mm256_set1_pd(h.re);
    let hi = _mm256_set1_pd(h.im);
    let mut j = 0;
    while j + 4 <= rho.len() {
        let a = _mm256_permute4x64_pd(_mm256_loadu_pd(rho.as_ptr().add(j).cast()), 0xd8);
        let b = _mm256_permute4x64_pd(_mm256_loadu_pd(rho.as_ptr().add(j + 2).cast()), 0xd8);
        let rr = _mm256_permute2f128_pd(a, b, 0x20);
        let ri = _mm256_permute2f128_pd(a, b, 0x31);
        let dr = _mm256_fmsub_pd(hr, rr, _mm256_mul_pd(hi, ri));
        let di = _mm256_fmadd_pd(hr, ri, _mm256_mul_pd(hi, rr));
        _mm256_storeu_pd(
            re.as_mut_ptr().add(j),
            _mm256_add_pd(_mm256_loadu_pd(re.as_ptr().add(j)), dr),
        );
        _mm256_storeu_pd(
            im.as_mut_ptr().add(j),
            _mm256_add_pd(_mm256_loadu_pd(im.as_ptr().add(j)), di),
        );
        j += 4;
    }
    scalar_row(h, &rho[j..], &mut re[j..], &mut im[j..]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn avx512_row(h: Complex64, rho: &[Complex64], re: &mut [f64], im: &mut [f64], fma: bool) {
    use std::arch::x86_64::*;
    let hr = _mm512_set1_pd(h.re);
    let hi = _mm512_set1_pd(h.im);
    let er = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    let ei = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    let mut j = 0;
    while j + 8 <= rho.len() {
        let a = _mm512_loadu_pd(rho.as_ptr().add(j).cast());
        let b = _mm512_loadu_pd(rho.as_ptr().add(j + 4).cast());
        let rr = _mm512_permutex2var_pd(a, er, b);
        let ri = _mm512_permutex2var_pd(a, ei, b);
        let (dr, di) = if fma {
            (
                _mm512_fmsub_pd(hr, rr, _mm512_mul_pd(hi, ri)),
                _mm512_fmadd_pd(hr, ri, _mm512_mul_pd(hi, rr)),
            )
        } else {
            (
                _mm512_sub_pd(_mm512_mul_pd(hr, rr), _mm512_mul_pd(hi, ri)),
                _mm512_add_pd(_mm512_mul_pd(hr, ri), _mm512_mul_pd(hi, rr)),
            )
        };
        _mm512_storeu_pd(
            re.as_mut_ptr().add(j),
            _mm512_add_pd(_mm512_loadu_pd(re.as_ptr().add(j)), dr),
        );
        _mm512_storeu_pd(
            im.as_mut_ptr().add(j),
            _mm512_add_pd(_mm512_loadu_pd(im.as_ptr().add(j)), di),
        );
        j += 8;
    }
    if fma {
        avx2_fma_row(h, &rho[j..], &mut re[j..], &mut im[j..]);
    } else {
        avx2_row(h, &rho[j..], &mut re[j..], &mut im[j..]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn finalize_packed(
    plan: &PreparedLindbladPlan,
    upper: &UpperTriLayout,
    h_upper: &[Complex64],
    rho: &[Complex64],
    scratch_re: &[f64],
    scratch_im: &[f64],
    diagonal: &mut [f64],
    out: &mut [f64],
) {
    use std::arch::x86_64::*;
    let n = plan.layout.n;
    for i in 0..n {
        diagonal[i] = h_upper[upper.index_unchecked(i, i)].re;
    }
    let half = _mm256_set1_pd(0.5);
    let mut cursor = n;
    for i in 0..n {
        let rate_i = plan.source_decay_rates[i];
        let mut pop = 2.0 * scratch_im[i * n + i] - rate_i * rho[i * n + i].re;
        for tr in &plan.incoming_transfers_by_target[i] {
            pop += tr.rate * rho[tr.source * n + tr.source].re;
        }
        out[i] = pop;
        let hii = _mm256_set1_pd(diagonal[i]);
        let ratei = _mm256_set1_pd(rate_i);
        let mut j = i + 1;
        while j + 4 <= n {
            let idx = _mm256_setr_epi64x(
                (j * n + i) as i64,
                ((j + 1) * n + i) as i64,
                ((j + 2) * n + i) as i64,
                ((j + 3) * n + i) as i64,
            );
            let cr = _mm256_add_pd(
                _mm256_loadu_pd(scratch_im.as_ptr().add(i * n + j)),
                _mm256_i64gather_pd(scratch_im.as_ptr(), idx, 8),
            );
            let ci = _mm256_sub_pd(
                _mm256_i64gather_pd(scratch_re.as_ptr(), idx, 8),
                _mm256_loadu_pd(scratch_re.as_ptr().add(i * n + j)),
            );
            let delta = _mm256_sub_pd(hii, _mm256_loadu_pd(diagonal.as_ptr().add(j)));
            let rate = _mm256_mul_pd(
                half,
                _mm256_add_pd(
                    ratei,
                    _mm256_loadu_pd(plan.source_decay_rates.as_ptr().add(j)),
                ),
            );
            let a =
                _mm256_permute4x64_pd(_mm256_loadu_pd(rho.as_ptr().add(i * n + j).cast()), 0xd8);
            let b = _mm256_permute4x64_pd(
                _mm256_loadu_pd(rho.as_ptr().add(i * n + j + 2).cast()),
                0xd8,
            );
            let rr = _mm256_permute2f128_pd(a, b, 0x20);
            let ri = _mm256_permute2f128_pd(a, b, 0x31);
            let re = _mm256_sub_pd(
                _mm256_add_pd(_mm256_mul_pd(delta, ri), cr),
                _mm256_mul_pd(rate, rr),
            );
            let im = _mm256_sub_pd(
                _mm256_sub_pd(ci, _mm256_mul_pd(delta, rr)),
                _mm256_mul_pd(rate, ri),
            );
            let lo = _mm256_unpacklo_pd(re, im);
            let hi = _mm256_unpackhi_pd(re, im);
            _mm256_storeu_pd(
                out.as_mut_ptr().add(cursor),
                _mm256_permute2f128_pd(lo, hi, 0x20),
            );
            _mm256_storeu_pd(
                out.as_mut_ptr().add(cursor + 4),
                _mm256_permute2f128_pd(lo, hi, 0x31),
            );
            cursor += 8;
            j += 4;
        }
        while j < n {
            let ij = i * n + j;
            let ji = j * n + i;
            let comm = Complex64::new(
                scratch_im[ij] + scratch_im[ji],
                scratch_re[ji] - scratch_re[ij],
            );
            let delta = diagonal[i] - diagonal[j];
            let mut value = Complex64::new(0.0, -delta) * rho[ij] + comm;
            value -= 0.5 * (rate_i + plan.source_decay_rates[j]) * rho[ij];
            out[cursor] = value.re;
            out[cursor + 1] = value.im;
            cursor += 2;
            j += 1;
        }
    }
}
