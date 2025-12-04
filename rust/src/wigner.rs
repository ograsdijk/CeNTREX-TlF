// src/wigner.rs

//! Simple pure-Rust Wigner 3j and 6j symbols using 2j / 2m integers.
//!
//! All j, m inputs are given as 2*j, 2*m (to allow half-integers
//! while staying with integers in the API).
//!
//! These implementations are fine for typical atomic/molecular
//! use (j up to a few tens). For very large j, factorials will
//! eventually overflow in f64 (~170! limit).

use std::cmp::{max, min};

fn fact(n: i32) -> f64 {
    if n < 0 {
        return 0.0;
    }
    if n == 0 || n == 1 {
        return 1.0;
    }
    let mut acc = 1.0f64;
    for k in 2..=n {
        acc *= k as f64;
    }
    acc
}

fn triangle_ok(tj1: i32, tj2: i32, tj3: i32) -> bool {
    // Triangle inequalities and parity:
    // |j1 - j2| <= j3 <= j1 + j2 and j1 + j2 + j3 integer.
    // In 2j notation: absolute values and sums are on the 2j's.
    if (tj1 + tj2 + tj3) % 2 != 0 {
        return false;
    }
    if tj3 < (tj1 - tj2).abs() {
        return false;
    }
    if tj3 > tj1 + tj2 {
        return false;
    }
    true
}

/// Racah triangle coefficient Δ(j1,j2,j3) in 2j notation.
fn delta_coeff(tj1: i32, tj2: i32, tj3: i32) -> f64 {
    if !triangle_ok(tj1, tj2, tj3) {
        return 0.0;
    }

    let a = (tj1 + tj2 - tj3) / 2;
    let b = (tj1 - tj2 + tj3) / 2;
    let c = (-tj1 + tj2 + tj3) / 2;
    let d = (tj1 + tj2 + tj3) / 2 + 1;

    if a < 0 || b < 0 || c < 0 || d <= 0 {
        return 0.0;
    }

    let num = fact(a) * fact(b) * fact(c);
    let den = fact(d);
    (num / den).sqrt()
}

/// Wigner 3j symbol in 2j / 2m notation.
///
/// Arguments: (2j1, 2j2, 2j3, 2m1, 2m2, 2m3)
pub fn wigner_3j(tj1: i32, tj2: i32, tj3: i32, tm1: i32, tm2: i32, tm3: i32) -> f64 {
    // Selection rules
    if tm1 + tm2 + tm3 != 0 {
        return 0.0;
    }
    if (tm1.abs() > tj1) || (tm2.abs() > tj2) || (tm3.abs() > tj3) {
        return 0.0;
    }
    if !triangle_ok(tj1, tj2, tj3) {
        return 0.0;
    }

    // Parity: j1 + j2 + j3 must be integer already checked by triangle_ok,
    // but also j1 - j2 - m3 exponent needs (tj1 - tj2 - tm3) even:
    if (tj1 - tj2 - tm3) % 2 != 0 {
        return 0.0;
    }

    // Prefactor Δ * sqrt of factorials
    let delta = delta_coeff(tj1, tj2, tj3);
    if delta == 0.0 {
        return 0.0;
    }

    let a1 = (tj1 + tm1) / 2;
    let a2 = (tj1 - tm1) / 2;
    let a3 = (tj2 + tm2) / 2;
    let a4 = (tj2 - tm2) / 2;
    let a5 = (tj3 + tm3) / 2;
    let a6 = (tj3 - tm3) / 2;

    if [a1, a2, a3, a4, a5, a6].iter().any(|&x| x < 0) {
        return 0.0;
    }

    let pref_sqrt = delta
        * (fact(a1)
            * fact(a2)
            * fact(a3)
            * fact(a4)
            * fact(a5)
            * fact(a6))
        .sqrt();

    // z-sum bounds in integer form (see standard 3j formula)
    let z_a1 = (tj1 + tj2 - tj3) / 2;           // = a1' in some notations
    let z_a2 = (tj1 - tm1) / 2;
    let z_a3 = (tj2 + tm2) / 2;
    let z_a4 = (tj3 - tj2 + tm1) / 2;
    let z_a5 = (tj3 - tj1 - tm2) / 2;

    let z_min = max(0, max(-z_a4, -z_a5));
    let z_max = min(z_a1, min(z_a2, z_a3));

    if z_min > z_max {
        return 0.0;
    }

    let mut sum = 0.0f64;
    for z in z_min..=z_max {
        let denom = fact(z)
            * fact(z_a1 - z)
            * fact(z_a2 - z)
            * fact(z_a3 - z)
            * fact(z_a4 + z)
            * fact(z_a5 + z);

        if denom == 0.0 {
            continue;
        }

        let term = if z % 2 == 0 { 1.0 } else { -1.0 } * (1.0 / denom);
        sum += term;
    }

    // Overall phase (-1)^{j1 - j2 - m3}
    let phase_exp = (tj1 - tj2 - tm3) / 2;
    let phase = if phase_exp % 2 == 0 { 1.0 } else { -1.0 };

    phase * pref_sqrt * sum
}

/// Wigner 6j symbol in 2j notation.
///
/// Arguments: (2j1, 2j2, 2j3, 2j4, 2j5, 2j6)
pub fn wigner_6j(tj1: i32, tj2: i32, tj3: i32, tj4: i32, tj5: i32, tj6: i32) -> f64 {
    // Triangle conditions on the four triples
    if !triangle_ok(tj1, tj2, tj3)
        || !triangle_ok(tj1, tj5, tj6)
        || !triangle_ok(tj4, tj2, tj6)
        || !triangle_ok(tj4, tj5, tj3)
    {
        return 0.0;
    }

    let delta123 = delta_coeff(tj1, tj2, tj3);
    let delta156 = delta_coeff(tj1, tj5, tj6);
    let delta426 = delta_coeff(tj4, tj2, tj6);
    let delta453 = delta_coeff(tj4, tj5, tj3);

    if [delta123, delta156, delta426, delta453]
        .iter()
        .any(|&d| d == 0.0)
    {
        return 0.0;
    }

    let pref = delta123 * delta156 * delta426 * delta453;

    // Racah sum over z
    let x1 = (tj1 + tj2 + tj3) / 2;
    let x2 = (tj1 + tj5 + tj6) / 2;
    let x3 = (tj4 + tj2 + tj6) / 2;
    let x4 = (tj4 + tj5 + tj3) / 2;

    let y1 = (tj1 + tj2 + tj4 + tj5) / 2;
    let y2 = (tj1 + tj3 + tj4 + tj6) / 2;
    let y3 = (tj2 + tj3 + tj5 + tj6) / 2;

    let z_min = max(max(x1, x2), max(x3, x4));
    let z_max = min(y1, min(y2, y3));

    if z_min > z_max {
        return 0.0;
    }

    let mut sum = 0.0f64;
    for z in z_min..=z_max {
        let num = fact(z + 1);
        let den = fact(z - x1)
            * fact(z - x2)
            * fact(z - x3)
            * fact(z - x4)
            * fact(y1 - z)
            * fact(y2 - z)
            * fact(y3 - z);

        if den == 0.0 {
            continue;
        }

        let sign = if z % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * num / den;
    }

    pref * sum
}

fn to_two_j(x: f64) -> i32 {
    let t = (2.0 * x).round();
    debug_assert!((t - 2.0 * x).abs() < 1e-6, "non half-integer j or m: {}", x);
    t as i32
}

/// Wigner 3j symbol for float inputs (wraps integer implementation).
pub fn wigner_3j_f(
    j1: f64,
    j2: f64,
    j3: f64,
    m1: f64,
    m2: f64,
    m3: f64,
) -> f64 {
    let tj1 = to_two_j(j1);
    let tj2 = to_two_j(j2);
    let tj3 = to_two_j(j3);
    let tm1 = to_two_j(m1);
    let tm2 = to_two_j(m2);
    let tm3 = to_two_j(m3);

    wigner_3j(tj1, tj2, tj3, tm1, tm2, tm3)
}

/// Wigner 6j symbol for float inputs (wraps integer implementation).
pub fn wigner_6j_f(
    j1: f64,
    j2: f64,
    j3: f64,
    j4: f64,
    j5: f64,
    j6: f64,
) -> f64 {
    let tj1 = to_two_j(j1);
    let tj2 = to_two_j(j2);
    let tj3 = to_two_j(j3);
    let tj4 = to_two_j(j4);
    let tj5 = to_two_j(j5);
    let tj6 = to_two_j(j6);

        wigner_6j(tj1, tj2, tj3, tj4, tj5, tj6)
}

/// Clebsch-Gordan coefficient <j1 m1 j2 m2 | j3 m3>
///
/// Arguments: (2j1, 2m1, 2j2, 2m2, 2j3, 2m3)
pub fn clebsch_gordan(tj1: i32, tm1: i32, tj2: i32, tm2: i32, tj3: i32, tm3: i32) -> f64 {
    // CG(j1,m1,j2,m2,j3,m3) = (-1)^(j1-j2+m3) * sqrt(2j3+1) * 3j(j1,j2,j3,m1,m2,-m3)

    // Check m1 + m2 = m3
    if tm1 + tm2 != tm3 {
        return 0.0;
    }

    let w3j = wigner_3j(tj1, tj2, tj3, tm1, tm2, -tm3);

    if w3j == 0.0 {
        return 0.0;
    }

    let phase_exp = (tj1 - tj2 + tm3) / 2;
    let phase = if phase_exp % 2 == 0 { 1.0 } else { -1.0 };

    let factor = ((tj3 + 1) as f64).sqrt();

    phase * factor * w3j
}
