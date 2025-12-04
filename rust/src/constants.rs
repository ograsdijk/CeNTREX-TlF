pub const A0: f64 = 0.529177210903e-10; // m
pub const DEBYE: f64 = 3.333333333333333e-30; // C·m
pub const DEBYE_HZ_V_CM: f64 = 503411.7791722602; // Hz/(V/cm)
pub const B_EPSILON: f64 = 6.689873e9; // Hz
pub const ALPHA: f64 = 45.0843e6; // Hz

#[derive(Clone, Copy, Debug)]
/// Constants for the X state Hamiltonian.
pub struct XConstants {
    /// Rotational constant (Hz)
    pub B_rot: f64,
    /// Hyperfine constant c1 (Hz)
    pub c1: f64,
    /// Hyperfine constant c2 (Hz)
    pub c2: f64,
    /// Hyperfine constant c3 (Hz)
    pub c3: f64,
    /// Hyperfine constant c4 (Hz)
    pub c4: f64,
    /// Rotational g-factor
    pub mu_J: f64,
    /// Thallium nuclear magnetic moment (Hz/G)
    pub mu_Tl: f64,
    /// Fluorine nuclear magnetic moment (Hz/G)
    pub mu_F: f64,
    /// Electric dipole moment (Hz/(V/cm))
    pub D_TlF: f64,
    /// Electric dipole moment (C·m)
    pub D: f64,
}

impl Default for XConstants {
    fn default() -> Self {
        XConstants {
            B_rot: B_EPSILON - ALPHA / 2.0,
            c1: 126030.0,
            c2: 17890.0,
            c3: 700.0,
            c4: -13300.0,
            mu_J: 35.0,
            mu_Tl: 1240.5,
            mu_F: 2003.63,
            D_TlF: 4.2282 * DEBYE_HZ_V_CM,
            D: 4.2282 * DEBYE,
        }
    }
}

#[derive(Clone, Debug)]
/// Constants for the B state Hamiltonian.
pub struct BConstants {
    /// Rotational constant (Hz)
    pub B_rot: f64,
    /// Centrifugal distortion constant (Hz)
    pub D_rot: f64,
    /// Higher order centrifugal distortion constant (Hz)
    pub H_const: f64,
    /// Tl hyperfine constant (Hz)
    pub h1_Tl: f64,
    /// F hyperfine constant (Hz)
    pub h1_F: f64,
    /// Lambda doubling constant (Hz)
    pub q: f64,
    /// Tl spin-rotation constant (Hz)
    pub c_Tl: f64,
    /// Tl spin-rotation constant (Hz)
    pub c1p_Tl: f64,
    /// Bohr magneton (Hz/G)
    pub mu_B: f64,
    /// Electron orbital g-factor
    pub gL: f64,
    /// Electron spin g-factor
    pub gS: f64,
    /// Electric dipole moment (Hz/(V/cm))
    pub mu_E: f64,
    /// Decay rate (Hz)
    pub Gamma: f64,
}
