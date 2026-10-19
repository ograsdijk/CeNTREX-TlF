// Isolated AVX2 row kernel used only by check_stage_reuse_simd.py.
// Complex values use num_complex's C-compatible pair-of-f64 layout.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_commutator_sparse_dense_upper_avx2_inner(
    pattern: &lindblad::plan::HermitianSparsePattern,
    h_sparse: &[Complex64],
    rho: &[Complex64],
    scratch: &mut [Complex64],
    upper_layout: &lindblad::layout::UpperTriLayout,
    drho_upper: &mut [Complex64],
) {
    use std::arch::x86_64::*;
    let n = pattern.n;
    let rho_f64 = rho.as_ptr().cast::<f64>();
    let scratch_f64 = scratch.as_mut_ptr().cast::<f64>();
    scratch.fill(Complex64::ZERO);
    let zero = _mm256_setzero_pd();
    for i in 0..n {
        for j0 in (0..n).step_by(2) {
            let mut acc = zero;
            for ptr in pattern.row_ptrs[i]..pattern.row_ptrs[i + 1] {
                let h = h_sparse[ptr];
                if h == Complex64::ZERO { continue; }
                let k = pattern.col_indices[ptr];
                if j0 + 1 < n {
                    let rv = _mm256_loadu_pd(rho_f64.add(2 * (k * n + j0)));
                    let swapped = _mm256_permute_pd(rv, 0b0101);
                    let hr = _mm256_set1_pd(h.re);
                    let hi = _mm256_set_pd(h.im, -h.im, h.im, -h.im);
                    acc = _mm256_add_pd(acc, _mm256_add_pd(_mm256_mul_pd(rv, hr), _mm256_mul_pd(swapped, hi)));
                } else {
                    let r = rho[k * n + j0];
                    acc = _mm256_add_pd(acc, _mm256_set_pd(
                        h.re * r.im + h.im * r.re,
                        h.re * r.re - h.im * r.im,
                        0.0, 0.0));
                }
            }
            for csc_ptr in pattern.col_ptrs[i]..pattern.col_ptrs[i + 1] {
                let k = pattern.row_indices_csc[csc_ptr];
                let h = h_sparse[pattern.csc_to_csr[csc_ptr]].conj();
                if h == Complex64::ZERO { continue; }
                if j0 + 1 < n {
                    let rv = _mm256_loadu_pd(rho_f64.add(2 * (k * n + j0)));
                    let swapped = _mm256_permute_pd(rv, 0b0101);
                    let hr = _mm256_set1_pd(h.re);
                    let hi = _mm256_set_pd(h.im, -h.im, h.im, -h.im);
                    acc = _mm256_add_pd(acc, _mm256_add_pd(_mm256_mul_pd(rv, hr), _mm256_mul_pd(swapped, hi)));
                } else {
                    let r = rho[k * n + j0];
                    acc = _mm256_add_pd(acc, _mm256_set_pd(
                        h.re * r.im + h.im * r.re,
                        h.re * r.re - h.im * r.im,
                        0.0, 0.0));
                }
            }
            if j0 + 1 < n {
                _mm256_storeu_pd(scratch_f64.add(2 * (i * n + j0)), acc);
            } else {
                let mut tail = [0.0; 4];
                _mm256_storeu_pd(tail.as_mut_ptr(), acc);
                scratch[i * n + j0] = Complex64::new(tail[2], tail[3]);
            }
        }
    }
    for i in 0..n {
        drho_upper[upper_layout.index_unchecked(i, i)] =
            Complex64::new(2.0 * scratch[i * n + i].im, 0.0);
        for j in (i + 1)..n {
            drho_upper[upper_layout.index_unchecked(i, j)] = -Complex64::I
                * (scratch[i * n + j] - scratch[j * n + i].conj());
        }
    }
}

fn add_commutator_sparse_dense_upper_avx2(
    pattern: &lindblad::plan::HermitianSparsePattern,
    h_sparse: &[Complex64],
    rho: &[Complex64],
    scratch: &mut [Complex64],
    upper_layout: &lindblad::layout::UpperTriLayout,
    drho_upper: &mut [Complex64],
) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        // SAFETY: runtime feature detection guards this target-feature function.
        unsafe { add_commutator_sparse_dense_upper_avx2_inner(
            pattern, h_sparse, rho, scratch, upper_layout, drho_upper
        ) }
        return;
    }
    let n = pattern.n;
    scratch.fill(Complex64::ZERO);
    for i in 0..n {
        for ptr in pattern.row_ptrs[i]..pattern.row_ptrs[i + 1] {
            let h = h_sparse[ptr];
            let k = pattern.col_indices[ptr];
            for j in 0..n { scratch[i * n + j] += h * rho[k * n + j]; }
        }
        for csc_ptr in pattern.col_ptrs[i]..pattern.col_ptrs[i + 1] {
            let k = pattern.row_indices_csc[csc_ptr];
            let h = h_sparse[pattern.csc_to_csr[csc_ptr]].conj();
            for j in 0..n { scratch[i * n + j] += h * rho[k * n + j]; }
        }
    }
    for i in 0..n {
        drho_upper[upper_layout.index_unchecked(i, i)] =
            Complex64::new(2.0 * scratch[i * n + i].im, 0.0);
        for j in (i + 1)..n {
            drho_upper[upper_layout.index_unchecked(i, j)] = -Complex64::I
                * (scratch[i * n + j] - scratch[j * n + i].conj());
        }
    }
}
