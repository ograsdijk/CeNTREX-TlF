// Benchmark flags: 1 combine diagonal phase+damping arithmetic (requires 2),
// 2 fuse decay into finalization, 4 directly write packed output, 8 SIMD.
pub fn benchmark_configure(w:&mut RhsWorkspace, flags:usize, _support:&[usize]) {
    assert!(flags&1==0 || flags&2!=0);
    w.bench_fusion_flags=flags;
}

fn bench_fused_dsu(plan:&PreparedLindbladPlan,w:&mut RhsWorkspace,packed:Option<&mut[f64]>) {
    let n=plan.layout.n;
    let flags=w.bench_fusion_flags;
    let fused_decay=flags&2!=0;
    let combine=flags&1!=0;
    let pack=flags&4!=0;
    #[cfg(target_arch="x86_64")]
    let simd=flags&8!=0 && std::is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch="x86_64"))]
    let simd=false;
    let mut packed=packed;
    w.scratch_re.fill(0.0);
    w.scratch_im.fill(0.0);
    let p=&plan.hamiltonian_sparse_pattern;
    // Keep the existing sparse traversal, diagonal split, and accumulation order.
    for i in 0..n {
        for ptr in p.row_ptrs[i]..p.row_ptrs[i+1] {
            let k=p.col_indices[ptr]; let h=w.h_sparse_values[ptr];
            if k==i || h==Complex64::ZERO {continue;}
            bench_row(h,&w.rho[k*n..(k+1)*n],&mut w.scratch_re[i*n..(i+1)*n],&mut w.scratch_im[i*n..(i+1)*n],simd);
        }
        for cp in p.col_ptrs[i]..p.col_ptrs[i+1] {
            let k=p.row_indices_csc[cp]; let h=w.h_sparse_values[p.csc_to_csr[cp]].conj();
            if h==Complex64::ZERO {continue;}
            bench_row(h,&w.rho[k*n..(k+1)*n],&mut w.scratch_re[i*n..(i+1)*n],&mut w.scratch_im[i*n..(i+1)*n],simd);
        }
    }
    let mut cursor=n;
    for i in 0..n {
        let mut diagonal=Complex64::new(2.0*w.scratch_im[i*n+i],0.0);
        let rate_i=plan.source_decay_rates[i];
        if fused_decay {
            diagonal-=rate_i*w.rho[i*n+i];
            for tr in &plan.incoming_transfers_by_target[i] {
                diagonal+=tr.rate*w.rho[tr.source*n+tr.source];
            }
        }
        if pack {packed.as_deref_mut().unwrap()[i]=diagonal.re;}
        else {w.drho_upper[w.upper_layout.index_unchecked(i,i)]=diagonal;}
        let h_ii=w.hamiltonian_upper[w.upper_layout.index_unchecked(i,i)].re;
        for j in i+1..n {
            let ij=i*n+j; let ji=j*n+i;
            let comm=Complex64::new(w.scratch_im[ij]+w.scratch_im[ji],w.scratch_re[ji]-w.scratch_re[ij]);
            let delta=h_ii-w.hamiltonian_upper[w.upper_layout.index_unchecked(j,j)].re;
            let rate=0.5*(rate_i+plan.source_decay_rates[j]);
            let value=if combine {
                Complex64::new(-rate,-delta)*w.rho[ij]+comm
            } else {
                let mut value=Complex64::new(0.0,-delta)*w.rho[ij]+comm;
                if fused_decay {value-=rate*w.rho[ij];}
                value
            };
            if pack {
                let out=packed.as_deref_mut().unwrap();
                out[cursor]=value.re;out[cursor+1]=value.im;cursor+=2;
            } else {w.drho_upper[w.upper_layout.index_unchecked(i,j)]=value;}
        }
    }
    if pack && !fused_decay {
        // Packing alone: retain a distinct decay pass, operating on packed data.
        let out=packed.as_deref_mut().unwrap();
        let mut cursor=n;
        for i in 0..n {
            let rate_i=plan.source_decay_rates[i];
            out[i]-=rate_i*w.rho[i*n+i].re;
            for j in i+1..n {
                let rate=0.5*(rate_i+plan.source_decay_rates[j]);
                out[cursor]-=rate*w.rho[i*n+j].re;
                out[cursor+1]-=rate*w.rho[i*n+j].im;
                cursor+=2;
            }
        }
        for (i,transfers) in plan.incoming_transfers_by_target.iter().enumerate() {
            for tr in transfers {out[i]+=tr.rate*w.rho[tr.source*n+tr.source].re;}
        }
    }
}
