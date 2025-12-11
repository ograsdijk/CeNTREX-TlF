use num_complex::Complex64;
use crate::states::{CoupledState, CoupledBasisState};
use crate::wigner::{wigner_3j_f, wigner_6j_f};

pub fn angular_part(
    pol_vec: [Complex64; 3],
    f: i32,
    mf: i32,
    fp: i32,
    mfp: i32,
) -> Complex64 {
    let q = mf - mfp;
    if q.abs() > 1 {
        return Complex64::new(0.0, 0.0);
    }

    let sqrt2 = 2.0_f64.sqrt();
    let i = Complex64::new(0.0, 1.0);

    let p_q = match q {
        1 => -(pol_vec[0] + i * pol_vec[1]) / sqrt2,
        -1 =>  (pol_vec[0] - i * pol_vec[1]) / sqrt2,
        0 =>   pol_vec[2],
        _ => unreachable!("q = {q}, but |q| ≤ 1 already checked"),
    };

    // (-1)^(F - mF)
    let exponent = f - mf;
    let phase = if exponent.rem_euclid(2) == 0 { 1.0 } else { -1.0 };

    let three_j = wigner_3j_f(
        f as f64,
        1.0,
        fp as f64,
        -mf as f64,
        q as f64,
        mfp as f64,
    );

    p_q * (phase * three_j)
}

/// Calculate electric dipole matrix element between coupled basis states.
///
/// Computes the electric dipole matrix element ⟨bra|D̂·ε|ket⟩ for transitions
/// between molecular eigenstates. Follows the formula in Oskari Timgren's
/// thesis (p. 131), using Wigner 3-j and 6-j symbols.
///
/// * `pol_vec` is the polarization vector in Cartesian basis [Ex, Ey, Ez].
/// * If `rme_only` is true, returns only the reduced matrix element
///   (no angular/polarization dependence).
///
/// Selection rules enforced:
///   |ΔΩ| ≤ 1, |ΔJ| ≤ 1, |ΔF| ≤ 1, and for full ME also |ΔmF| ≤ 1.
///
/// NOTE on quantum-number representation:
///   In `CoupledBasisState`,
///   - `J, F, mF, Omega` are stored as physical integers (0, ±1, ±2, …)
///   - `I1, I2, F1` are stored as *twice* their physical values (2I₁, 2I₂, 2F₁).
pub fn ed_me_coupled(
    bra: &CoupledBasisState,
    ket: &CoupledBasisState,
    pol_vec: [Complex64; 3], // [Ex, Ey, Ez]
    rme_only: bool,
) -> Complex64 {
    // ---------- selection-rule early exits ----------
    if (bra.Omega - ket.Omega).abs() > 1 {
        return Complex64::new(0.0, 0.0);
    }
    if (bra.J - ket.J).abs() > 1 {
        return Complex64::new(0.0, 0.0);
    }
    if (bra.F - ket.F).abs() > 1 {
        return Complex64::new(0.0, 0.0);
    }
    if !rme_only && (bra.mF - ket.mF).abs() > 1 {
        return Complex64::new(0.0, 0.0);
    }

    // ---------- bra quantum numbers ----------
    let f = bra.F;
    let mf = bra.mF;
    let j = bra.J;
    let f1_twice = bra.F1; // 2*F1 (physical)
    let i1_twice = bra.I1; // 2*I1
    let i2_twice = bra.I2; // 2*I2
    let omega = bra.Omega;

    // ---------- ket quantum numbers ----------
    let fp = ket.F;
    let mfp = ket.mF;
    let jp = ket.J;
    let f1p_twice = ket.F1; // 2*F1'
    let omegap = ket.Omega;

    // q = Ω − Ω'
    let q = omega - omegap;
    if q.abs() > 1 {
        return Complex64::new(0.0, 0.0);
    }

    // ---------- phase: (-1)^(F1' + F1 + F' + I1 + I2 − Ω) ----------
    //
    // In Python those were physical numbers. Here:
    //   F1_phys  = F1_twice / 2
    //   F1p_phys = F1p_twice / 2
    //   I1_phys  = I1_twice / 2
    //   I2_phys  = I2_twice / 2
    //
    // So
    //   e = F1p + F1 + F' + I1 + I2 − Ω (physical)
    //   2e = F1p_twice + F1_twice + 2*F' + I1_twice + I2_twice − 2*Ω
    //
    // We only care about e mod 2 for (-1)^e.
    let two_e: i32 =
        f1p_twice + f1_twice + 2 * fp + i1_twice + i2_twice - 2 * omega;
    let e_parity = (two_e / 2).rem_euclid(2);
    let phase = if e_parity == 0 { 1.0 } else { -1.0 };

    // ---------- prefactor sqrt[(2J+1)(2J'+1)(2F1+1)(2F1'+1)(2F+1)(2F'+1)] ----------
    // F1,F1' stored as 2*F1_phys ⇒ (2F1_phys + 1) = F1_twice + 1, etc.
    let prefactor = (
        (2 * j + 1) as f64
        * (2 * jp + 1) as f64
        * (f1_twice + 1) as f64
        * (f1p_twice + 1) as f64
        * (2 * f + 1) as f64
        * (2 * fp + 1) as f64
    )
        .sqrt();

    // ---------- Wigner 6j symbols ----------
    //
    // Python: sixj_f(F1p, Fp, I2, F, F1, 1)
    // Here wigner_6j_f takes physical j's as f64 and internally does 2j.
    let f1_phys = f1_twice as f64 / 2.0;
    let f1p_phys = f1p_twice as f64 / 2.0;
    let i1_phys = i1_twice as f64 / 2.0;
    let i2_phys = i2_twice as f64 / 2.0;

    let sixj1 = wigner_6j_f(
        f1p_phys,
        fp as f64,
        i2_phys,
        f as f64,
        f1_phys,
        1.0,
    );

    // Python: sixj_f(Jp, F1p, I1, F1, J, 1)
    let sixj2 = wigner_6j_f(
        jp as f64,
        f1p_phys,
        i1_phys,
        f1_phys,
        j as f64,
        1.0,
    );

    // ---------- Wigner 3j symbol ----------
    //
    // Python: threej_f(J, 1, Jp, -Omega, q, Omegap)
    let threej = wigner_3j_f(
        j as f64,
        1.0,
        jp as f64,
        -omega as f64,
        q as f64,
        omegap as f64,
    );

    let me_scalar = phase * prefactor * sixj1 * sixj2 * threej;

    if me_scalar == 0.0 {
        return Complex64::new(0.0, 0.0);
    }

    let mut me = Complex64::new(me_scalar, 0.0);

    // ---------- include polarization/angular dependence ----------
    if !rme_only {
        let ang = angular_part(pol_vec, f, mf, fp, mfp);
        me *= ang;
    }

    me
}

/// Electric dipole matrix element between mixed (superposition) states.
///
/// This is the Rust version of `generate_ED_ME_mixed_state`
/// but with:
///   - `pol_vec` REQUIRED (no defaults)
///   - NO polarization normalization
///
/// Computes  ⟨bra | D·ε | ket⟩ for mixed quantum states.
pub fn generate_ed_me_mixed_state(
    bra: &CoupledState,
    ket: &CoupledState,
    pol_vec: [Complex64; 3],
    reduced: bool,
) -> Complex64 {
    let mut me = Complex64::new(0.0, 0.0);

    // ME = Σ_{a,b} amp_bra* · amp_ket · ED_ME_coupled(basis_bra, basis_ket)
    for (amp_bra, basis_bra) in bra.terms.iter() {
        let amp_bra_conj = amp_bra.conj();

        for (amp_ket, basis_ket) in ket.terms.iter() {
            if (basis_bra.J - basis_ket.J).abs() > 1 {
                continue;
            }

            let me_basis = ed_me_coupled(basis_bra, basis_ket, pol_vec, reduced);

            me += amp_bra_conj * (*amp_ket) * me_basis;
        }
    }

    me
}

/// Generate optical coupling matrix for transitions between quantum states.
///
/// Now takes ground and excited **indices** into `qn`, instead of separate
/// ground_states / excited_states slices.
pub fn generate_coupling_matrix(
    qn: &[CoupledState],
    ground_indices: &[usize],
    excited_indices: &[usize],
    pol_vec: [Complex64; 3],
    reduced: bool,
) -> Vec<Vec<Complex64>> {
    let n = qn.len();
    let mut h = vec![vec![Complex64::new(0.0, 0.0); n]; n];

    for &i in ground_indices {
        let ground_state = &qn[i];

        for &j in excited_indices {
            let excited_state = &qn[j];

            let me = generate_ed_me_mixed_state(
                excited_state,
                ground_state,
                pol_vec,
                reduced,
            );

            h[i][j] = me;
            if me != Complex64::new(0.0, 0.0) {
                h[j][i] = me.conj();
            }
        }
    }

    h
}