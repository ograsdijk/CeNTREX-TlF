// Flags 16: FMA products; 32: AVX-512 commutator; 64: AVX2 finalization.
#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2,fma")]
unsafe fn ext_fma_row(h:Complex64,rho:&[Complex64],re:&mut[f64],im:&mut[f64]) {
    use std::arch::x86_64::*;
    let hr=_mm256_set1_pd(h.re);let hi=_mm256_set1_pd(h.im);
    let mut j=0;
    while j+4<=rho.len() {
        let a=_mm256_permute4x64_pd(_mm256_loadu_pd(rho.as_ptr().add(j).cast()),0xd8);
        let b=_mm256_permute4x64_pd(_mm256_loadu_pd(rho.as_ptr().add(j+2).cast()),0xd8);
        let rr=_mm256_permute2f128_pd(a,b,0x20);
        let ri=_mm256_permute2f128_pd(a,b,0x31);
        let dr=_mm256_fmsub_pd(hr,rr,_mm256_mul_pd(hi,ri));
        let di=_mm256_fmadd_pd(hr,ri,_mm256_mul_pd(hi,rr));
        _mm256_storeu_pd(re.as_mut_ptr().add(j),_mm256_add_pd(_mm256_loadu_pd(re.as_ptr().add(j)),dr));
        _mm256_storeu_pd(im.as_mut_ptr().add(j),_mm256_add_pd(_mm256_loadu_pd(im.as_ptr().add(j)),di));
        j+=4;
    }
    bench_scalar_row(h,&rho[j..],&mut re[j..],&mut im[j..]);
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx512f,avx2,fma")]
unsafe fn ext_wide_row(h:Complex64,rho:&[Complex64],re:&mut[f64],im:&mut[f64],fma:bool) {
    use std::arch::x86_64::*;
    let hr=_mm512_set1_pd(h.re);let hi=_mm512_set1_pd(h.im);
    let er=_mm512_setr_epi64(0,2,4,6,8,10,12,14);
    let ei=_mm512_setr_epi64(1,3,5,7,9,11,13,15);
    let mut j=0;
    while j+8<=rho.len() {
        let a=_mm512_loadu_pd(rho.as_ptr().add(j).cast());
        let b=_mm512_loadu_pd(rho.as_ptr().add(j+4).cast());
        let rr=_mm512_permutex2var_pd(a,er,b);
        let ri=_mm512_permutex2var_pd(a,ei,b);
        let (dr,di)=if fma {
            (_mm512_fmsub_pd(hr,rr,_mm512_mul_pd(hi,ri)),_mm512_fmadd_pd(hr,ri,_mm512_mul_pd(hi,rr)))
        } else {
            (_mm512_sub_pd(_mm512_mul_pd(hr,rr),_mm512_mul_pd(hi,ri)),
             _mm512_add_pd(_mm512_mul_pd(hr,ri),_mm512_mul_pd(hi,rr)))
        };
        _mm512_storeu_pd(re.as_mut_ptr().add(j),_mm512_add_pd(_mm512_loadu_pd(re.as_ptr().add(j)),dr));
        _mm512_storeu_pd(im.as_mut_ptr().add(j),_mm512_add_pd(_mm512_loadu_pd(im.as_ptr().add(j)),di));
        j+=8;
    }
    if fma {ext_fma_row(h,&rho[j..],&mut re[j..],&mut im[j..]);}
    else {bench_avx_row(h,&rho[j..],&mut re[j..],&mut im[j..]);}
}

#[inline]
fn ext_dispatch(h:Complex64,rho:&[Complex64],re:&mut[f64],im:&mut[f64],simd:bool,flags:usize) {
    // ISA support is checked once at benchmark configuration, outside timing.
    #[cfg(target_arch="x86_64")]
    unsafe {
        if flags&32!=0 {ext_wide_row(h,rho,re,im,flags&16!=0);return;}
        if flags&16!=0 {ext_fma_row(h,rho,re,im);return;}
    }
    bench_row(h,rho,re,im,simd);
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2")]
unsafe fn ext_finalize(plan:&PreparedLindbladPlan,w:&mut RhsWorkspace,out:&mut[f64]) {
    use std::arch::x86_64::*;
    let n=plan.layout.n;
    for i in 0..n {
        w.bench_hdiag[i]=w.hamiltonian_upper[w.upper_layout.index_unchecked(i,i)].re;
    }
    let half=_mm256_set1_pd(0.5);
    let mut cursor=n;
    for i in 0..n {
        let rate_i=plan.source_decay_rates[i];
        let mut pop=2.0*w.scratch_im[i*n+i]-rate_i*w.rho[i*n+i].re;
        for tr in &plan.incoming_transfers_by_target[i] {pop+=tr.rate*w.rho[tr.source*n+tr.source].re;}
        out[i]=pop;
        let hii=_mm256_set1_pd(w.bench_hdiag[i]);
        let ratei=_mm256_set1_pd(rate_i);
        let mut j=i+1;
        while j+4<=n {
            let idx=_mm256_setr_epi64x((j*n+i) as i64,((j+1)*n+i) as i64,((j+2)*n+i) as i64,((j+3)*n+i) as i64);
            let cr=_mm256_add_pd(_mm256_loadu_pd(w.scratch_im.as_ptr().add(i*n+j)),_mm256_i64gather_pd(w.scratch_im.as_ptr(),idx,8));
            let ci=_mm256_sub_pd(_mm256_i64gather_pd(w.scratch_re.as_ptr(),idx,8),_mm256_loadu_pd(w.scratch_re.as_ptr().add(i*n+j)));
            let delta=_mm256_sub_pd(hii,_mm256_loadu_pd(w.bench_hdiag.as_ptr().add(j)));
            let rate=_mm256_mul_pd(half,_mm256_add_pd(ratei,_mm256_loadu_pd(plan.source_decay_rates.as_ptr().add(j))));
            let a=_mm256_permute4x64_pd(_mm256_loadu_pd(w.rho.as_ptr().add(i*n+j).cast()),0xd8);
            let b=_mm256_permute4x64_pd(_mm256_loadu_pd(w.rho.as_ptr().add(i*n+j+2).cast()),0xd8);
            let rr=_mm256_permute2f128_pd(a,b,0x20);
            let ri=_mm256_permute2f128_pd(a,b,0x31);
            let re=_mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(delta,ri),cr),_mm256_mul_pd(rate,rr));
            let im=_mm256_sub_pd(_mm256_sub_pd(ci,_mm256_mul_pd(delta,rr)),_mm256_mul_pd(rate,ri));
            let lo=_mm256_unpacklo_pd(re,im);let hi=_mm256_unpackhi_pd(re,im);
            _mm256_storeu_pd(out.as_mut_ptr().add(cursor),_mm256_permute2f128_pd(lo,hi,0x20));
            _mm256_storeu_pd(out.as_mut_ptr().add(cursor+4),_mm256_permute2f128_pd(lo,hi,0x31));
            cursor+=8;j+=4;
        }
        while j<n {
            let ij=i*n+j;let ji=j*n+i;
            let comm=Complex64::new(w.scratch_im[ij]+w.scratch_im[ji],w.scratch_re[ji]-w.scratch_re[ij]);
            let delta=w.bench_hdiag[i]-w.bench_hdiag[j];
            let mut value=Complex64::new(0.0,-delta)*w.rho[ij]+comm;
            value-=0.5*(rate_i+plan.source_decay_rates[j])*w.rho[ij];
            out[cursor]=value.re;out[cursor+1]=value.im;
            cursor+=2;j+=1;
        }
    }
}
