use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods,PyList, PyTuple};
use pyo3::Py;
use pyo3::PyResult;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use num_complex::Complex64;

mod states;
mod quantum_operators;
mod constants;
mod X_uncoupled;
mod B_coupled;
mod generate_hamiltonian;
mod coupling;
pub mod wigner;

use states::{UncoupledBasisState, CoupledBasisState, CoupledState};
use constants::{XConstants, BConstants};
use generate_hamiltonian::{generate_uncoupled_hamiltonian_X, generate_coupled_hamiltonian_B};
use coupling::generate_coupling_matrix;
use std::collections::HashMap;


use wigner::{wigner_3j_f, wigner_6j_f};

#[pyfunction(signature = (j1, j2, j3, m1, m2, m3))]
/// Calculate the Wigner 3j symbol using Rust.
///
/// Args:
///     j1 (float): Angular momentum 1.
///     j2 (float): Angular momentum 2.
///     j3 (float): Angular momentum 3.
///     m1 (float): Projection 1.
///     m2 (float): Projection 2.
///     m3 (float): Projection 3.
///
/// Returns:
///     float: The value of the Wigner 3j symbol.
fn wigner_3j_py(j1: f64, j2: f64, j3: f64, m1: f64, m2: f64, m3: f64) -> f64 {
    wigner_3j_f(j1, j2, j3, m1, m2, m3)
}

#[pyfunction(signature = (j1, j2, j3, j4, j5, j6))]
/// Calculate the Wigner 6j symbol using Rust.
///
/// Args:
///     j1 (float): Angular momentum 1.
///     j2 (float): Angular momentum 2.
///     j3 (float): Angular momentum 3.
///     j4 (float): Angular momentum 4.
///     j5 (float): Angular momentum 5.
///     j6 (float): Angular momentum 6.
///
/// Returns:
///     float: The value of the Wigner 6j symbol.
fn wigner_6j_py(j1: f64, j2: f64, j3: f64, j4: f64, j5: f64, j6: f64) -> f64 {
    wigner_6j_f(j1, j2, j3, j4, j5, j6)
}

#[pyfunction(signature = (states, constants))]
/// Generate the uncoupled X state Hamiltonian for the supplied basis states using Rust.
///
/// Args:
///     states (Sequence[UncoupledBasisState]): Array of uncoupled basis states.
///     constants (XConstants): X state molecular constants.
///
/// Returns:
///     HamiltonianUncoupledX: Dataclass containing all X state Hamiltonian matrix terms.
fn generate_uncoupled_hamiltonian_X_py<'py>(
    py: Python<'py>,
    states: Vec<Bound<'py, PyAny>>,
    constants: &Bound<'py, PyAny>
) -> PyResult<Bound<'py, PyAny>> {
    // Convert Python states to Rust UncoupledBasisState
    let rust_states: Vec<UncoupledBasisState> = states.iter().map(|s| {
        let J: i32 = s.getattr("J")?.extract()?;
        let mJ: i32 = s.getattr("mJ")?.extract()?;
        let I1: f64 = s.getattr("I1")?.extract()?;
        let m1: f64 = s.getattr("m1")?.extract()?;
        let I2: f64 = s.getattr("I2")?.extract()?;
        let m2: f64 = s.getattr("m2")?.extract()?;
        let Omega: i32 = s.getattr("Omega")?.extract()?;
        let P: i32 = s.getattr("P")?.extract()?;

        Ok(UncoupledBasisState {
            J,
            mJ,
            I1: (I1 * 2.0).round() as i32,
            m1: (m1 * 2.0).round() as i32,
            I2: (I2 * 2.0).round() as i32,
            m2: (m2 * 2.0).round() as i32,
            Omega,
            parity: P as i8
        })
    }).collect::<PyResult<Vec<_>>>()?;

    // Convert Python constants to Rust XConstants
    let rust_constants = XConstants {
        B_rot: constants.getattr("B_rot")?.extract()?,
        c1: constants.getattr("c1")?.extract()?,
        c2: constants.getattr("c2")?.extract()?,
        c3: constants.getattr("c3")?.extract()?,
        c4: constants.getattr("c4")?.extract()?,
        mu_J: constants.getattr("μ_J")?.extract()?,
        mu_Tl: constants.getattr("μ_Tl")?.extract()?,
        mu_F: constants.getattr("μ_F")?.extract()?,
        D_TlF: constants.getattr("D_TlF")?.extract()?,
        D: constants.getattr("D")?.extract()?,
    };

    let result = generate_uncoupled_hamiltonian_X(&rust_states, &rust_constants);
    let n = rust_states.len();

    let to_numpy = |vec: Vec<Complex64>| -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let array = PyArray1::from_vec(py, vec);
        array.reshape((n, n))
    };

    let ham_cls = py.import("centrex_tlf.hamiltonian")?.getattr("HamiltonianUncoupledX")?;

    let args = (
        to_numpy(result.Hff)?,
        to_numpy(result.HSx)?,
        to_numpy(result.HSy)?,
        to_numpy(result.HSz)?,
        to_numpy(result.HZx)?,
        to_numpy(result.HZy)?,
        to_numpy(result.HZz)?,
    );

    ham_cls.call1(args)
}

#[pyfunction(signature = (states, constants))]
/// Generate the coupled B state Hamiltonian for the supplied basis states using Rust.
///
/// Args:
///     states (Sequence[CoupledBasisState]): Array of coupled basis states.
///     constants (BConstants): B state molecular constants.
///
/// Returns:
///     HamiltonianCoupledBOmega: Dataclass containing all B state Hamiltonian matrix terms.
fn generate_coupled_hamiltonian_B_py<'py>(
    py: Python<'py>,
    states: Vec<Bound<'py, PyAny>>,
    constants: &Bound<'py, PyAny>
) -> PyResult<Bound<'py, PyAny>> {
    let qn: Vec<CoupledBasisState> = states.iter().map(|s| {
        let J: i32 = s.getattr("J")?.extract()?;
        let F: i32 = s.getattr("F")?.extract()?;
        let mF: i32 = s.getattr("mF")?.extract()?;
        let I1: i32 = (s.getattr("I1")?.extract::<f64>()? * 2.0).round() as i32;
        let I2: i32 = (s.getattr("I2")?.extract::<f64>()? * 2.0).round() as i32;
        let F1: i32 = (s.getattr("F1")?.extract::<f64>()? * 2.0).round() as i32;
        let Omega: i32 = s.getattr("Omega")?.extract()?;

        let P_obj = s.getattr("P")?;
        let P: Option<i8> = if P_obj.is_none() {
            None
        } else {
            Some(P_obj.extract::<i8>()?)
        };

        Ok::<CoupledBasisState, PyErr>(CoupledBasisState {
            J, F, mF, I1, I2, F1, Omega, P,
            electronic_state: states::ElectronicState::B
        })
    }).collect::<Result<Vec<_>, _>>()?;

    // Manual extraction of BConstants
    let constants_obj = constants;
    let constants = BConstants {
        B_rot: constants_obj.getattr("B_rot")?.extract()?,
        D_rot: constants_obj.getattr("D_rot")?.extract()?,
        H_const: constants_obj.getattr("H_const")?.extract()?,
        h1_Tl: constants_obj.getattr("h1_Tl")?.extract()?,
        h1_F: constants_obj.getattr("h1_F")?.extract()?,
        q: constants_obj.getattr("q")?.extract()?,
        c_Tl: constants_obj.getattr("c_Tl")?.extract()?,
        c1p_Tl: constants_obj.getattr("c1p_Tl")?.extract()?,
        mu_B: constants_obj.getattr("μ_B")?.extract()?,
        gL: constants_obj.getattr("gL")?.extract()?,
        gS: constants_obj.getattr("gS")?.extract()?,
        mu_E: constants_obj.getattr("μ_E")?.extract()?,
        Gamma: constants_obj.getattr("Γ")?.extract()?,
    };

    let hamiltonian = generate_coupled_hamiltonian_B(&qn, &constants);

    let ham_cls = py.import("centrex_tlf.hamiltonian")?.getattr("HamiltonianCoupledBOmega")?;

    let n = qn.len();
    let shape = (n, n);

    let to_pyarray = |vec: Vec<Complex64>| -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let array = PyArray1::from_vec(py, vec);
        array.reshape(shape)
    };

    let args = (
        to_pyarray(hamiltonian.Hrot)?,
        to_pyarray(hamiltonian.H_mhf_Tl)?,
        to_pyarray(hamiltonian.H_mhf_F)?,
        to_pyarray(hamiltonian.H_LD)?,
        to_pyarray(hamiltonian.H_cp1_Tl)?,
        to_pyarray(hamiltonian.H_c_Tl)?,
        to_pyarray(hamiltonian.HSx)?,
        to_pyarray(hamiltonian.HSy)?,
        to_pyarray(hamiltonian.HSz)?,
        to_pyarray(hamiltonian.HZx)?,
        to_pyarray(hamiltonian.HZy)?,
        to_pyarray(hamiltonian.HZz)?,
    );

    ham_cls.call1(args)
}

#[pyfunction(signature = (basis1, basis2))]
/// Generate transformation matrix between two quantum state bases using Rust.
///
/// Computes the transformation matrix S where S[i,j] = <basis1[i]|basis2[j]>.
///
/// Args:
///     basis1 (Sequence[BasisState] | npt.NDArray): First basis.
///     basis2 (Sequence[BasisState] | npt.NDArray): Second basis.
///
/// Returns:
///     npt.NDArray[np.complex128]: Transformation matrix S.
fn generate_transform_matrix_py<'py>(
    py: Python<'py>,
    basis1: Vec<Bound<'py, PyAny>>,
    basis2: Vec<Bound<'py, PyAny>>
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let parse_state = |s: &Bound<'py, PyAny>| -> PyResult<states::BasisStateEnum> {
        let is_coupled: bool = s.getattr("isCoupled")?.extract()?;
        if is_coupled {
            let J: i32 = s.getattr("J")?.extract()?;
            let F: i32 = s.getattr("F")?.extract()?;
            let mF: i32 = s.getattr("mF")?.extract()?;
            let I1: i32 = (s.getattr("I1")?.extract::<f64>()? * 2.0).round() as i32;
            let I2: i32 = (s.getattr("I2")?.extract::<f64>()? * 2.0).round() as i32;
            let F1: i32 = (s.getattr("F1")?.extract::<f64>()? * 2.0).round() as i32;
            let Omega: i32 = s.getattr("Omega")?.extract()?;
            let P_obj = s.getattr("P")?;
            let P: Option<i8> = if P_obj.is_none() { None } else { Some(P_obj.extract::<i8>()?) };

            let es_obj = s.getattr("electronic_state")?;
            let es_name: String = es_obj.getattr("name")?.extract()?;
            let electronic_state = match es_name.as_str() {
                "X" => states::ElectronicState::X,
                "B" => states::ElectronicState::B,
                _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Unknown electronic state")),
            };

            Ok(states::BasisStateEnum::Coupled(states::CoupledBasisState {
                J, F, mF, I1, I2, F1, Omega, P, electronic_state
            }))
        } else {
            let J: i32 = s.getattr("J")?.extract()?;
            let mJ: i32 = s.getattr("mJ")?.extract()?;
            let I1: i32 = (s.getattr("I1")?.extract::<f64>()? * 2.0).round() as i32;
            let m1: i32 = (s.getattr("m1")?.extract::<f64>()? * 2.0).round() as i32;
            let I2: i32 = (s.getattr("I2")?.extract::<f64>()? * 2.0).round() as i32;
            let m2: i32 = (s.getattr("m2")?.extract::<f64>()? * 2.0).round() as i32;
            let Omega: i32 = s.getattr("Omega")?.extract()?;
            let P_obj = s.getattr("P")?;
            let parity: i8 = if P_obj.is_none() { 0 } else { P_obj.extract::<i8>()? };

            Ok(states::BasisStateEnum::Uncoupled(states::UncoupledBasisState {
                J, mJ, I1, m1, I2, m2, Omega, parity
            }))
        }
    };

    let b1: Vec<states::BasisStateEnum> = basis1.iter().map(|s| parse_state(s)).collect::<PyResult<Vec<_>>>()?;
    let b2: Vec<states::BasisStateEnum> = basis2.iter().map(|s| parse_state(s)).collect::<PyResult<Vec<_>>>()?;

    let n1 = b1.len();
    let n2 = b2.len();
    let mut matrix = Vec::with_capacity(n1 * n2);

    for s1 in &b1 {
        for s2 in &b2 {
            matrix.push(s1.inner_product(s2));
        }
    }

    let array = PyArray1::from_vec(py, matrix);
    array.reshape((n1, n2))
}

#[pyfunction(signature = (QN, ground_states, excited_states, pol_vec, reduced=false))]
/// Generate optical coupling matrix for transitions between quantum states.
///
/// Python wrapper around Rust `generate_coupling_matrix`.
fn generate_coupling_matrix_py<'py>(
    py: Python<'py>,
    QN: Vec<Bound<'py, PyAny>>,
    ground_states: Vec<Bound<'py, PyAny>>,
    excited_states: Vec<Bound<'py, PyAny>>,
    pol_vec: &Bound<'py, PyAny>,  // accept ndarray or list
    reduced: bool,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    // --- pol_vec: Python sequence -> [Complex64; 3] (no normalization) ---
    let pol_list: Vec<Complex64> = pol_vec.extract()?;
    if pol_list.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "pol_vec must have length 3",
        ));
    }
    let pol = [pol_list[0], pol_list[1], pol_list[2]];

    // --- helper: Python CoupledState -> Rust CoupledState ---
    let parse_coupled_state = |s: &Bound<'py, PyAny>| -> PyResult<CoupledState> {
        // Python side: CoupledState.data is iterable of (amp, basis) pairs
        let terms_obj = s.getattr("data")?;
        let terms_list = terms_obj.downcast::<PyList>()?;

        let mut terms = Vec::with_capacity(terms_list.len());

        for term in terms_list.iter() {
            // each term is (amp, basis_state)
            let tup = term.downcast::<PyTuple>()?;
            let amp: Complex64 = tup.get_item(0)?.extract()?;
            let basis_py = tup.get_item(1)?;

            // CoupledBasisState fields
            let J: i32 = basis_py.getattr("J")?.extract()?;
            let F: i32 = basis_py.getattr("F")?.extract()?;
            let mF: i32 = basis_py.getattr("mF")?.extract()?;

            // I1, I2, F1 stored as 2×physical in Rust
            let I1: i32 = (basis_py.getattr("I1")?.extract::<f64>()? * 2.0).round() as i32;
            let I2: i32 = (basis_py.getattr("I2")?.extract::<f64>()? * 2.0).round() as i32;
            let F1: i32 = (basis_py.getattr("F1")?.extract::<f64>()? * 2.0).round() as i32;

            let Omega: i32 = basis_py.getattr("Omega")?.extract()?;

            let P_obj = basis_py.getattr("P")?;
            let P: Option<i8> = if P_obj.is_none() {
                None
            } else {
                Some(P_obj.extract::<i8>()?)
            };

            let es_obj = basis_py.getattr("electronic_state")?;
            let es_name: String = es_obj.getattr("name")?.extract()?;
            let electronic_state = match es_name.as_str() {
                "X" => states::ElectronicState::X,
                "B" => states::ElectronicState::B,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Unknown electronic state",
                    ))
                }
            };

            let basis = CoupledBasisState {
                J,
                F,
                mF,
                I1,
                I2,
                F1,
                Omega,
                P,
                electronic_state,
            };

            terms.push((amp, basis));
        }

        Ok(CoupledState::new(terms))
    };

    // --- convert QN (Python) -> rust_qn (Rust) ---
    let rust_qn: Vec<CoupledState> = QN
        .iter()
        .map(|s| parse_coupled_state(s))
        .collect::<PyResult<_>>()?;

    // --- build identity-based mapping: Python object -> index in QN ---
    let mut ptr_to_idx: HashMap<usize, usize> = HashMap::with_capacity(QN.len());
    for (i, obj) in QN.iter().enumerate() {
        let ptr = obj.as_ptr() as usize;
        ptr_to_idx.insert(ptr, i);
    }

    // --- compute ground_indices and excited_indices using Python identity ---
    let mut ground_indices = Vec::with_capacity(ground_states.len());
    for gs in ground_states.iter() {
        let ptr = gs.as_ptr() as usize;
        let idx = ptr_to_idx.get(&ptr).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "ground_state not found in QN",
            )
        })?;
        ground_indices.push(*idx);
    }

    let mut excited_indices = Vec::with_capacity(excited_states.len());
    for es in excited_states.iter() {
        let ptr = es.as_ptr() as usize;
        let idx = ptr_to_idx.get(&ptr).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "excited_state not found in QN",
            )
        })?;
        excited_indices.push(*idx);
    }

    // --- call core Rust function (no pol normalization) ---
    let H = generate_coupling_matrix(
        &rust_qn,
        &ground_indices,
        &excited_indices,
        pol,
        reduced,
    );

    // --- convert Vec<Vec<Complex64>> -> numpy.ndarray (n×n) ---
    let array = PyArray2::from_vec2(py, &H)?;
    Ok(array)
}


#[pymodule]
fn centrex_tlf_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_uncoupled_hamiltonian_X_py, m)?)?;
    m.add_function(wrap_pyfunction!(generate_coupled_hamiltonian_B_py, m)?)?;
    m.add_function(wrap_pyfunction!(wigner_3j_py, m)?)?;
    m.add_function(wrap_pyfunction!(wigner_6j_py, m)?)?;
    m.add_function(wrap_pyfunction!(generate_transform_matrix_py, m)?)?;
    m.add_function(wrap_pyfunction!(generate_coupling_matrix_py, m)?)?;
    Ok(())
}
