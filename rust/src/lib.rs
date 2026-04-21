use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyTuple};
use pyo3::PyResult;

mod b_coupled;
mod constants;
mod coupling;
mod generate_hamiltonian;
mod lindblad;
mod quantum_operators;
mod states;
pub mod wigner;
mod x_uncoupled;

use constants::{BConstants, XConstants};
use coupling::generate_coupling_matrix;
use generate_hamiltonian::{generate_coupled_hamiltonian_b, generate_uncoupled_hamiltonian_x};
use states::{CoupledBasisState, CoupledState, UncoupledBasisState};

use wigner::{wigner_3j_f, wigner_6j_f};

fn parse_uncoupled_state(s: &Bound<'_, PyAny>) -> PyResult<UncoupledBasisState> {
    let j: i32 = s.getattr("J")?.extract()?;
    let mj: i32 = s.getattr("mJ")?.extract()?;
    let i1: i32 = (s.getattr("I1")?.extract::<f64>()? * 2.0).round() as i32;
    let m1: i32 = (s.getattr("m1")?.extract::<f64>()? * 2.0).round() as i32;
    let i2: i32 = (s.getattr("I2")?.extract::<f64>()? * 2.0).round() as i32;
    let m2: i32 = (s.getattr("m2")?.extract::<f64>()? * 2.0).round() as i32;
    let omega: i32 = s.getattr("Omega")?.extract()?;
    let p_obj = s.getattr("P")?;
    let parity: i8 = if p_obj.is_none() {
        0
    } else {
        p_obj.extract::<i8>()?
    };

    Ok(UncoupledBasisState {
        j,
        mj,
        i1,
        m1,
        i2,
        m2,
        omega,
        parity,
    })
}

fn parse_electronic_state(s: &Bound<'_, PyAny>) -> PyResult<states::ElectronicState> {
    let es_obj = s.getattr("electronic_state")?;
    let es_name: String = es_obj.getattr("name")?.extract()?;
    match es_name.as_str() {
        "X" => Ok(states::ElectronicState::X),
        "B" => Ok(states::ElectronicState::B),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unknown electronic state",
        )),
    }
}

fn parse_optional_parity(s: &Bound<'_, PyAny>) -> PyResult<Option<i8>> {
    let p_obj = s.getattr("P")?;
    if p_obj.is_none() {
        Ok(None)
    } else {
        Ok(Some(p_obj.extract::<i8>()?))
    }
}

fn parse_coupled_basis_state(s: &Bound<'_, PyAny>) -> PyResult<CoupledBasisState> {
    Ok(CoupledBasisState {
        j: s.getattr("J")?.extract()?,
        f: s.getattr("F")?.extract()?,
        mf: s.getattr("mF")?.extract()?,
        i1: (s.getattr("I1")?.extract::<f64>()? * 2.0).round() as i32,
        i2: (s.getattr("I2")?.extract::<f64>()? * 2.0).round() as i32,
        f1: (s.getattr("F1")?.extract::<f64>()? * 2.0).round() as i32,
        omega: s.getattr("Omega")?.extract()?,
        parity: parse_optional_parity(s)?,
        electronic_state: parse_electronic_state(s)?,
    })
}

fn parse_basis_state_enum(s: &Bound<'_, PyAny>) -> PyResult<states::BasisStateEnum> {
    let is_coupled: bool = s.getattr("isCoupled")?.extract()?;
    if is_coupled {
        Ok(states::BasisStateEnum::Coupled(parse_coupled_basis_state(
            s,
        )?))
    } else {
        Ok(states::BasisStateEnum::Uncoupled(parse_uncoupled_state(s)?))
    }
}

fn parse_coupled_state(s: &Bound<'_, PyAny>) -> PyResult<CoupledState> {
    let terms_obj = s.getattr("data")?;
    let terms_list: &Bound<'_, PyList> = terms_obj.cast()?;
    let mut terms = Vec::with_capacity(terms_list.len());

    for term in terms_list.iter() {
        let tup: &Bound<'_, PyTuple> = term.cast()?;
        let amp: Complex64 = tup.get_item(0)?.extract()?;
        let basis_py = tup.get_item(1)?;
        let basis = parse_coupled_basis_state(&basis_py)?;
        terms.push((amp, basis));
    }

    Ok(CoupledState::new(terms))
}

fn parse_x_constants(constants: &Bound<'_, PyAny>) -> PyResult<XConstants> {
    Ok(XConstants {
        b_rot: constants.getattr("B_rot")?.extract()?,
        c1: constants.getattr("c1")?.extract()?,
        c2: constants.getattr("c2")?.extract()?,
        c3: constants.getattr("c3")?.extract()?,
        c4: constants.getattr("c4")?.extract()?,
        mu_j: constants.getattr("μ_J")?.extract()?,
        mu_tl: constants.getattr("μ_Tl")?.extract()?,
        mu_f: constants.getattr("μ_F")?.extract()?,
        d_tlf: constants.getattr("D_TlF")?.extract()?,
        d: constants.getattr("D")?.extract()?,
    })
}

fn parse_b_constants(constants: &Bound<'_, PyAny>) -> PyResult<BConstants> {
    Ok(BConstants {
        b_rot: constants.getattr("B_rot")?.extract()?,
        d_rot: constants.getattr("D_rot")?.extract()?,
        h_const: constants.getattr("H_const")?.extract()?,
        h1_tl: constants.getattr("h1_Tl")?.extract()?,
        h1_f: constants.getattr("h1_F")?.extract()?,
        q: constants.getattr("q")?.extract()?,
        c_tl: constants.getattr("c_Tl")?.extract()?,
        c1p_tl: constants.getattr("c1p_Tl")?.extract()?,
        mu_b: constants.getattr("μ_B")?.extract()?,
        gl: constants.getattr("gL")?.extract()?,
        gs: constants.getattr("gS")?.extract()?,
        mu_e: constants.getattr("μ_E")?.extract()?,
        gamma: constants.getattr("Γ")?.extract()?,
    })
}

fn parse_pol_vec(pol_vec: &Bound<'_, PyAny>) -> PyResult<[Complex64; 3]> {
    let pol_list: Vec<Complex64> = pol_vec.extract()?;
    if pol_list.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "pol_vec must have length 3",
        ));
    }
    Ok([pol_list[0], pol_list[1], pol_list[2]])
}

#[pyfunction(signature = (j1, j2, j3, m1, m2, m3))]
fn wigner_3j_py(j1: f64, j2: f64, j3: f64, m1: f64, m2: f64, m3: f64) -> f64 {
    wigner_3j_f(j1, j2, j3, m1, m2, m3)
}

#[pyfunction(signature = (j1, j2, j3, j4, j5, j6))]
fn wigner_6j_py(j1: f64, j2: f64, j3: f64, j4: f64, j5: f64, j6: f64) -> f64 {
    wigner_6j_f(j1, j2, j3, j4, j5, j6)
}

#[pyfunction(signature = (states, constants))]
fn generate_uncoupled_hamiltonian_X_py<'py>(
    py: Python<'py>,
    states: Vec<Bound<'py, PyAny>>,
    constants: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let rust_states: Vec<UncoupledBasisState> = states
        .iter()
        .map(parse_uncoupled_state)
        .collect::<PyResult<Vec<_>>>()?;
    let rust_constants = parse_x_constants(constants)?;

    let result = generate_uncoupled_hamiltonian_x(&rust_states, &rust_constants);
    let n = rust_states.len();

    let to_numpy = |vec: Vec<Complex64>| -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let array = PyArray1::from_vec(py, vec);
        array.reshape((n, n))
    };

    let ham_cls = py
        .import("centrex_tlf.hamiltonian")?
        .getattr("HamiltonianUncoupledX")?;
    ham_cls.call1((
        to_numpy(result.h_ff)?,
        to_numpy(result.h_sx)?,
        to_numpy(result.h_sy)?,
        to_numpy(result.h_sz)?,
        to_numpy(result.h_zx)?,
        to_numpy(result.h_zy)?,
        to_numpy(result.h_zz)?,
    ))
}

#[pyfunction(signature = (states, constants))]
fn generate_coupled_hamiltonian_B_py<'py>(
    py: Python<'py>,
    states: Vec<Bound<'py, PyAny>>,
    constants: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let qn: Vec<CoupledBasisState> = states
        .iter()
        .map(|s| parse_coupled_basis_state(s))
        .collect::<PyResult<Vec<_>>>()?;
    let constants = parse_b_constants(constants)?;

    let hamiltonian = generate_coupled_hamiltonian_b(&qn, &constants);

    let n = qn.len();
    let to_pyarray = |vec: Vec<Complex64>| -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let array = PyArray1::from_vec(py, vec);
        array.reshape((n, n))
    };

    let ham_cls = py
        .import("centrex_tlf.hamiltonian")?
        .getattr("HamiltonianCoupledBOmega")?;
    ham_cls.call1((
        to_pyarray(hamiltonian.h_rot)?,
        to_pyarray(hamiltonian.h_mhf_tl)?,
        to_pyarray(hamiltonian.h_mhf_f)?,
        to_pyarray(hamiltonian.h_ld)?,
        to_pyarray(hamiltonian.h_cp1_tl)?,
        to_pyarray(hamiltonian.h_c_tl)?,
        to_pyarray(hamiltonian.h_sx)?,
        to_pyarray(hamiltonian.h_sy)?,
        to_pyarray(hamiltonian.h_sz)?,
        to_pyarray(hamiltonian.h_zx)?,
        to_pyarray(hamiltonian.h_zy)?,
        to_pyarray(hamiltonian.h_zz)?,
    ))
}

#[pyfunction(signature = (basis1, basis2))]
fn generate_transform_matrix_py<'py>(
    py: Python<'py>,
    basis1: Vec<Bound<'py, PyAny>>,
    basis2: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let b1: Vec<states::BasisStateEnum> = basis1
        .iter()
        .map(parse_basis_state_enum)
        .collect::<PyResult<Vec<_>>>()?;
    let b2: Vec<states::BasisStateEnum> = basis2
        .iter()
        .map(parse_basis_state_enum)
        .collect::<PyResult<Vec<_>>>()?;

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

#[pyfunction(signature = (qn, ground_indices, excited_indices, pol_vec, reduced=false))]
fn generate_coupling_matrix_py<'py>(
    py: Python<'py>,
    qn: Vec<Bound<'py, PyAny>>,
    ground_indices: Vec<usize>,
    excited_indices: Vec<usize>,
    pol_vec: &Bound<'py, PyAny>,
    reduced: bool,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let pol = parse_pol_vec(pol_vec)?;

    let rust_qn: Vec<CoupledState> = qn
        .iter()
        .map(|s| parse_coupled_state(s))
        .collect::<PyResult<_>>()?;

    let h = generate_coupling_matrix(&rust_qn, &ground_indices, &excited_indices, pol, reduced);

    let n = rust_qn.len();
    let array = PyArray1::from_vec(py, h);
    array.reshape((n, n))
}

#[pymodule]
fn centrex_tlf_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_uncoupled_hamiltonian_X_py, m)?)?;
    m.add_function(wrap_pyfunction!(generate_coupled_hamiltonian_B_py, m)?)?;
    m.add_function(wrap_pyfunction!(wigner_3j_py, m)?)?;
    m.add_function(wrap_pyfunction!(wigner_6j_py, m)?)?;
    m.add_function(wrap_pyfunction!(generate_transform_matrix_py, m)?)?;
    m.add_function(wrap_pyfunction!(generate_coupling_matrix_py, m)?)?;
    lindblad::register_python_api(m)?;
    Ok(())
}
