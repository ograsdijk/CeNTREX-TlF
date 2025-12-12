pub const A0: f64 = 0.529177210903e-10; // m
pub const DEBYE: f64 = 3.333333333333333e-30; // C·m
pub const DEBYE_HZ_V_CM: f64 = 503411.7791722602; // Hz/(V/cm)
pub const B_EPSILON: f64 = 6.689873e9; // Hz
pub const ALPHA: f64 = 45.0843e6; // Hz

#[derive(Clone, Copy, Debug)]
/// Constants for the X state Hamiltonian.
pub struct XConstants {
    /// Rotational constant (Hz)
    pub b_rot: f64,
    /// Hyperfine constant c1 (Hz)
    pub c1: f64,
    /// Hyperfine constant c2 (Hz)
    pub c2: f64,
    /// Hyperfine constant c3 (Hz)
    pub c3: f64,
    /// Hyperfine constant c4 (Hz)
    pub c4: f64,
    /// Rotational g-factor
    pub mu_j: f64,
    /// Thallium nuclear magnetic moment (Hz/G)
    pub mu_tl: f64,
    /// Fluorine nuclear magnetic moment (Hz/G)
    pub mu_f: f64,
    /// Electric dipole moment (Hz/(V/cm))
    pub d_tlf: f64,
    /// Electric dipole moment (C·m)
    pub d: f64,
}

impl Default for XConstants {
    fn default() -> Self {
        XConstants {
            b_rot: B_EPSILON - ALPHA / 2.0,
            c1: 126030.0,
            c2: 17890.0,
            c3: 700.0,
            c4: -13300.0,
            mu_j: 35.0,
            mu_tl: 1240.5,
            mu_f: 2003.63,
            d_tlf: 4.2282 * DEBYE_HZ_V_CM,
            d: 4.2282 * DEBYE,
        }
    }
}

#[derive(Clone, Debug)]
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
    /// Decay rate (Hz)
    pub gamma: f64,
}
