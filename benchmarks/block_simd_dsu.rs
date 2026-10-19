// Benchmark-only D+S+U kernel. Keeps the sparse outer traversal, diagonal
// splitting and separate real/imaginary accumulation used by the current RHS.
pub fn benchmark_configure(w: &mut RhsWorkspace, flags: usize, support: &[usize]) {
    w.bench_flags = flags;
    let n = w.upper_layout.n;
    w.bench_support = vec![false; n * n];
    for &index in support { w.bench_support[index] = true; }
    w.bench_runs = (0..n).map(|i| {
        let mut runs = Vec::new();
        let mut j = 0;
        while j < n {
            if !w.bench_support[i*n+j] { j += 1; continue; }
            let start = j;
            while j < n && w.bench_support[i*n+j] { j += 1; }
            runs.push((start,j));
        }
        runs
    }).collect();
}

#[inline]
fn bench_scalar_row(h: Complex64, rho: &[Complex64], re: &mut [f64], im: &mut [f64]) {
    for j in 0..rho.len() {
        let r = rho[j];
        re[j] += h.re*r.re - h.im*r.im;
        im[j] += h.re*r.im + h.im*r.re;
    }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn bench_avx_row(h: Complex64, rho: &[Complex64], re: &mut [f64], im: &mut [f64]) {
    use std::arch::x86_64::*;
    let hr = _mm256_set1_pd(h.re);
    let hi = _mm256_set1_pd(h.im);
    let mut j = 0;
    while j+4 <= rho.len() {
        let a = _mm256_permute4x64_pd(_mm256_loadu_pd(rho.as_ptr().add(j).cast()), 0xd8);
        let b = _mm256_permute4x64_pd(_mm256_loadu_pd(rho.as_ptr().add(j+2).cast()), 0xd8);
        let rr = _mm256_permute2f128_pd(a,b,0x20);
        let ri = _mm256_permute2f128_pd(a,b,0x31);
        let dr = _mm256_sub_pd(_mm256_mul_pd(hr,rr),_mm256_mul_pd(hi,ri));
        let di = _mm256_add_pd(_mm256_mul_pd(hr,ri),_mm256_mul_pd(hi,rr));
        _mm256_storeu_pd(re.as_mut_ptr().add(j),_mm256_add_pd(_mm256_loadu_pd(re.as_ptr().add(j)),dr));
        _mm256_storeu_pd(im.as_mut_ptr().add(j),_mm256_add_pd(_mm256_loadu_pd(im.as_ptr().add(j)),di));
        j += 4;
    }
    bench_scalar_row(h,&rho[j..],&mut re[j..],&mut im[j..]);
}

#[inline]
fn bench_row(h: Complex64, rho: &[Complex64], re: &mut [f64], im: &mut [f64], simd: bool) {
    #[cfg(target_arch="x86_64")]
    if simd {
        // SAFETY: bench_simd_dsu guards entry using runtime AVX2 detection.
        unsafe { bench_avx_row(h,rho,re,im); }
        return;
    }
    bench_scalar_row(h,rho,re,im);
}

fn bench_simd_dsu(
    pattern: &HermitianSparsePattern, h_sparse: &[Complex64], h_upper: &[Complex64],
    rho: &[Complex64], scratch_re: &mut [f64], scratch_im: &mut [f64],
    upper: &UpperTriLayout, out: &mut [Complex64], split: bool,
    flags: usize, runs: &[Vec<(usize,usize)>], support: &[bool],
) {
    let n = pattern.n;
    let block = flags & 1 != 0;
    #[cfg(target_arch="x86_64")]
    let simd = flags & 2 != 0 && std::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch="x86_64"))]
    let simd = false;
    scratch_re.fill(0.0);
    scratch_im.fill(0.0);
    for i in 0..n {
        for ptr in pattern.row_ptrs[i]..pattern.row_ptrs[i+1] {
            let h = h_sparse[ptr];
            let k = pattern.col_indices[ptr];
            if (split && k == i) || h == Complex64::ZERO { continue; }
            if block {
                for &(a,b) in &runs[i] {
                    bench_row(h,&rho[k*n+a..k*n+b],&mut scratch_re[i*n+a..i*n+b],&mut scratch_im[i*n+a..i*n+b],simd);
                }
            } else {
                bench_row(h,&rho[k*n..(k+1)*n],&mut scratch_re[i*n..(i+1)*n],&mut scratch_im[i*n..(i+1)*n],simd);
            }
        }
        for cp in pattern.col_ptrs[i]..pattern.col_ptrs[i+1] {
            let k = pattern.row_indices_csc[cp];
            let h = h_sparse[pattern.csc_to_csr[cp]].conj();
            if h == Complex64::ZERO { continue; }
            if block {
                for &(a,b) in &runs[i] {
                    bench_row(h,&rho[k*n+a..k*n+b],&mut scratch_re[i*n+a..i*n+b],&mut scratch_im[i*n+a..i*n+b],simd);
                }
            } else {
                bench_row(h,&rho[k*n..(k+1)*n],&mut scratch_re[i*n..(i+1)*n],&mut scratch_im[i*n..(i+1)*n],simd);
            }
        }
    }
    for i in 0..n {
        out[upper.index_unchecked(i,i)] = Complex64::new(2.0*scratch_im[i*n+i],0.0);
        for j in i+1..n {
            if block && !support[i*n+j] { continue; }
            let ij=i*n+j; let ji=j*n+i;
            let comm = Complex64::new(scratch_im[ij]+scratch_im[ji],scratch_re[ji]-scratch_re[ij]);
            let diag = if split {
                Complex64::new(0.0,-(h_upper[upper.index_unchecked(i,i)].re-h_upper[upper.index_unchecked(j,j)].re))*rho[ij]
            } else {Complex64::ZERO};
            out[upper.index_unchecked(i,j)] = diag+comm;
        }
    }
}
