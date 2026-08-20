// Fundamental constants and unit conversions. Python derives every value in
// this block from `scipy.constants` at import time (see the corresponding
// definitions in centrex_tlf/constants/constants.py); here they are frozen
// literals. `test_rust_constants_match_scipy_derived_python` in
// tests/test_constants.py pins the two sides together, so a CODATA revision in
// a newer SciPy fails loudly instead of silently desynchronising the backends.
pub const DEBYE: f64 = 3.33564095198152e-30; // C·m; 1e-21 / c
pub const DEBYE_HZ_V_CM: f64 = 503411.6567542709; // Hz/(V/cm); Debye * 100 / h
pub const BOHR_MAGNETON_HZ_G: f64 = 1399624.4917100002; // Hz/G
pub const ELEMENTARY_CHARGE: f64 = 1.602176634e-19; // C; scipy.constants.e
pub const BOHR_RADIUS: f64 = 5.29177210544e-11; // m; scipy.constants.value("Bohr radius")

// X-B transition dipole moment. Kept in the same multiplication order as
// Python's `ED_XtB = 0.315 * cst.e * a0` so the two are bit-identical in f64.
pub const ED_XTB: f64 = 0.315 * ELEMENTARY_CHARGE * BOHR_RADIUS; // C·m

// X-state Dunham coefficients Y_kl for 205Tl19F, mirroring the NIST values
// used in Python (centrex_tlf/constants/constants.py). Source: NIST Diatomic
// Spectral Database, TlF table.
pub const Y01_X: f64 = 6689.8736e6; // Hz
pub const Y11_X: f64 = -45.0843e6; // Hz
pub const Y21_X: f64 = 0.0942e6; // Hz
pub const Y02_X: f64 = -5.84e3; // Hz

// v=0 rotational constants derived from the Dunham coefficients above. Do
// not hard-code a different B0/D0 value independently of Y01_X/Y11_X/...
pub const B0_X: f64 = Y01_X + Y11_X / 2.0 + Y21_X / 4.0; // Hz
pub const D0_X: f64 = -Y02_X; // Hz

// X-state v=0 permanent electric dipole moment mu_0 = 4.2282(8) D, mirroring
// the NIST value used in Python (see MU_X_V0_D in constants.py). Only v=0 is
// needed in Rust; the v=1/v=2 values are not currently used here.
pub const MU_X_V0_D: f64 = 4.2282; // D

// Hunter et al., Phys. Rev. A 85, 012511 (2012), arXiv:1110.3748: measured
// B^3Pi_1(v'=0) lifetime tau = 99(9) ns, mirroring B_LIFETIME in Python
// (centrex_tlf/constants/constants.py).
pub const B_LIFETIME: f64 = 99e-9; // s

#[derive(Clone, Copy, Debug)]
/// Constants for the X state Hamiltonian.
pub struct XConstants {
    /// v=0 rotational constant (Hz)
    pub b_rot: f64,
    /// v=0 quartic centrifugal-distortion constant (Hz)
    pub d_rot: f64,
    /// Tl nuclear spin-rotation constant (Hz); NIST C_Tl = 126.03(12) kHz
    pub c1: f64,
    /// F nuclear spin-rotation constant (Hz); NIST C_F = 17.89(15) kHz
    pub c2: f64,
    /// Tensor nuclear spin-spin constant (Hz), in this package's operator
    /// normalization: NIST tabulates 3.50(15) kHz in a different
    /// normalization; this value (700 Hz) is that NIST value divided by 5.
    /// This is a convention difference, not a discrepant measurement.
    pub c3: f64,
    /// Scalar nuclear spin-spin constant (Hz); NIST c4 = -13.30(72) kHz
    pub c4: f64,
    /// Rotational magnetic moment (Hz/G), uncoupled-basis Zeeman term
    pub mu_j: f64,
    /// Thallium nuclear magnetic moment (Hz/G), uncoupled-basis Zeeman term
    pub mu_tl: f64,
    /// Fluorine nuclear magnetic moment (Hz/G), uncoupled-basis Zeeman term
    pub mu_f: f64,
    /// Body-fixed permanent electric dipole moment (Hz/(V/cm)), v=0 value
    /// MU_X_V0_D = 4.2282(8) D
    pub d_tlf: f64,
    /// Body-fixed permanent electric dipole moment (C·m), v=0 value
    /// MU_X_V0_D = 4.2282(8) D
    pub d: f64,
}

impl Default for XConstants {
    fn default() -> Self {
        XConstants {
            b_rot: B0_X,
            d_rot: D0_X,
            c1: 126030.0,
            c2: 17890.0,
            c3: 700.0,
            c4: -13300.0,
            mu_j: 35.0,
            mu_tl: 1240.5,
            mu_f: 2003.63,
            d_tlf: MU_X_V0_D * DEBYE_HZ_V_CM,
            d: MU_X_V0_D * DEBYE,
        }
    }
}

#[derive(Clone, Copy, Debug)]
/// Constants for the B state Hamiltonian.
pub struct BConstants {
    /// Rotational constant (Hz)
    pub b_rot: f64,
    /// Centrifugal distortion constant (Hz)
    pub d_rot: f64,
    /// Higher order centrifugal distortion constant (Hz)
    pub h_const: f64,
    /// Tl hyperfine constant (Hz)
    pub h1_tl: f64,
    /// F hyperfine constant (Hz)
    pub h1_f: f64,
    /// Lambda doubling constant (Hz)
    pub q: f64,
    /// Tl spin-rotation constant (Hz)
    pub c_tl: f64,
    /// Tl spin-rotation constant (Hz)
    pub c1p_tl: f64,
    /// Bohr magneton (Hz/G)
    pub mu_b: f64,
    /// Electron orbital g-factor
    pub gl: f64,
    /// Electron spin g-factor
    pub gs: f64,
    /// Electric dipole moment (Hz/(V/cm))
    pub mu_e: f64,
    /// Excited-state population decay rate (s^-1), derived directly from the
    /// measured B-state lifetime as gamma = 1 / B_LIFETIME (tau = 99(9) ns,
    /// Hunter et al., Phys. Rev. A 85, 012511 (2012)). The equivalent
    /// spectroscopic linewidth is gamma / (2*pi) ~= 1.608 MHz.
    pub gamma: f64,
}

impl Default for BConstants {
    fn default() -> Self {
        BConstants {
            b_rot: 6687.879e6,
            d_rot: 0.010869e6,
            h_const: -8.1e-2,
            h1_tl: 28789e6,
            h1_f: 861e6,
            q: 2.423e6,
            c_tl: -7.83e6,
            c1p_tl: 11.17e6,
            mu_b: BOHR_MAGNETON_HZ_G,
            gl: 1.0,
            gs: 2.0,
            mu_e: 2.28 * DEBYE_HZ_V_CM,
            gamma: 1.0 / B_LIFETIME,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_b_lifetime_value() {
        assert_eq!(B_LIFETIME, 99e-9);
    }

    #[test]
    fn test_bconstants_gamma_derived_from_lifetime() {
        let constants = BConstants::default();
        assert_eq!(constants.gamma, 1.0 / B_LIFETIME);
    }

    #[test]
    fn test_bconstants_gamma_equivalent_linewidth() {
        let constants = BConstants::default();
        let linewidth_mhz = constants.gamma / (2.0 * std::f64::consts::PI) / 1e6;
        assert!(
            (linewidth_mhz - 1.6076).abs() < 1e-3,
            "expected ~1.6076 MHz, got {linewidth_mhz}"
        );
    }
}
