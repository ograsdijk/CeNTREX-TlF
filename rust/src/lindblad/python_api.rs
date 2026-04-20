use crate::lindblad::plan::{parse_plan_payload, PreparedLindbladPlan};
use crate::lindblad::rhs::{
    build_packed_jacobian_sparse, build_split_jacobian_sparse, rhs_matrix_into,
    rhs_matrix_into_with_profile, rhs_packed, rhs_packed_into_with_profile,
    rhs_split_into_with_profile, ExecutionMode, RhsProfileStats, RhsWorkspace,
};
use crate::lindblad::solver_bdf::{solve_bdf, BdfSolverOptions};
use crate::lindblad::solver_ode::{solve_dopri5, OdeSolverOptions};
use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use std::cell::{Cell, RefCell};

#[pyclass(module = "centrex_tlf.centrex_tlf_rust", unsendable)]
pub struct LindbladRhsEvaluator {
    plan: PreparedLindbladPlan,
    mode: ExecutionMode,
    workspace: RefCell<RhsWorkspace>,
    profiling_enabled: Cell<bool>,
    profile: RefCell<RhsProfileStats>,
}

#[pymethods]
impl LindbladRhsEvaluator {
    #[pyo3(signature = (enabled = true))]
    pub fn enable_profile_py(&self, enabled: bool) {
        self.profiling_enabled.set(enabled);
    }

    pub fn reset_profile_py(&self) {
        *self.profile.borrow_mut() = RhsProfileStats::default();
    }

    pub fn profile_summary_py<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        let stats = self.profile.borrow().clone();
        let summary = PyDict::new(py);
        summary.set_item("enabled", self.profiling_enabled.get())?;
        summary.set_item("calls", stats.calls)?;
        summary.set_item("total_seconds", stats.total_seconds)?;
        summary.set_item("unpack_seconds", stats.unpack_seconds)?;
        summary.set_item("parameter_eval_seconds", stats.parameter_eval_seconds)?;
        summary.set_item("hamiltonian_fill_seconds", stats.hamiltonian_fill_seconds)?;
        summary.set_item("commutator_seconds", stats.commutator_seconds)?;
        summary.set_item("dissipator_seconds", stats.dissipator_seconds)?;
        summary.set_item("pack_seconds", stats.pack_seconds)?;
        if stats.calls > 0 {
            summary.set_item(
                "average_total_seconds",
                stats.total_seconds / stats.calls as f64,
            )?;
        } else {
            summary.set_item("average_total_seconds", 0.0)?;
        }
        Ok(summary.unbind())
    }

    pub fn rhs_packed_py<'py>(
        &self,
        py: Python<'py>,
        packed_state: PyReadonlyArray1<'py, f64>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let state = packed_state.as_slice().map_err(PyValueError::new_err)?;
        let mut out = vec![0.0; self.plan.layout.packed_len()];
        let mut workspace = self.workspace.borrow_mut();
        if self.profiling_enabled.get() {
            let mut profile = self.profile.borrow_mut();
            rhs_packed_into_with_profile(
                &self.plan,
                state,
                t,
                self.mode,
                &mut workspace,
                out.as_mut_slice(),
                Some(&mut profile),
            )
            .map_err(PyValueError::new_err)?;
        } else {
            rhs_packed_into_with_profile(
                &self.plan,
                state,
                t,
                self.mode,
                &mut workspace,
                out.as_mut_slice(),
                None,
            )
            .map_err(PyValueError::new_err)?;
        }
        Ok(PyArray1::from_vec(py, out))
    }

    pub fn rhs_matrix_py<'py>(
        &self,
        py: Python<'py>,
        matrix_state: PyReadonlyArray1<'py, Complex64>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
        let state = matrix_state.as_slice().map_err(PyValueError::new_err)?;
        let n = self.plan.n_states();
        if state.len() != n * n {
            return Err(PyValueError::new_err(format!(
                "expected full matrix state length {}, got {}",
                n * n,
                state.len()
            )));
        }
        let mut out = vec![Complex64::new(0.0, 0.0); n * n];
        let mut workspace = self.workspace.borrow_mut();
        if self.profiling_enabled.get() {
            let mut profile = self.profile.borrow_mut();
            rhs_matrix_into_with_profile(
                &self.plan,
                state,
                t,
                self.mode,
                &mut workspace,
                out.as_mut_slice(),
                Some(&mut profile),
            )
            .map_err(PyValueError::new_err)?;
        } else {
            rhs_matrix_into(
                &self.plan,
                state,
                t,
                self.mode,
                &mut workspace,
                out.as_mut_slice(),
            )
            .map_err(PyValueError::new_err)?;
        }
        Ok(PyArray1::from_vec(py, out))
    }

    pub fn rhs_split_py<'py>(
        &self,
        py: Python<'py>,
        split_state: PyReadonlyArray1<'py, f64>,
        t: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let state = split_state.as_slice().map_err(PyValueError::new_err)?;
        let n = self.plan.n_states() * self.plan.n_states();
        if state.len() != 2 * n {
            return Err(PyValueError::new_err(format!(
                "expected split-real state length {}, got {}",
                2 * n,
                state.len()
            )));
        }
        let mut out = vec![0.0; 2 * n];
        let mut workspace = self.workspace.borrow_mut();
        if self.profiling_enabled.get() {
            let mut profile = self.profile.borrow_mut();
            rhs_split_into_with_profile(
                &self.plan,
                state,
                t,
                self.mode,
                &mut workspace,
                out.as_mut_slice(),
                Some(&mut profile),
            )
            .map_err(PyValueError::new_err)?;
        } else {
            rhs_split_into_with_profile(
                &self.plan,
                state,
                t,
                self.mode,
                &mut workspace,
                out.as_mut_slice(),
                None,
            )
            .map_err(PyValueError::new_err)?;
        }
        Ok(PyArray1::from_vec(py, out))
    }

    #[pyo3(signature = (t, tol = 0.0))]
    pub fn jacobian_split_sparse_py<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        tol: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let mut workspace = self.workspace.borrow_mut();
        let (rows, cols, values) =
            build_split_jacobian_sparse(&self.plan, t, self.mode, &mut workspace, tol)
                .map_err(PyValueError::new_err)?;
        Ok((
            PyArray1::from_vec(py, rows),
            PyArray1::from_vec(py, cols),
            PyArray1::from_vec(py, values),
        ))
    }

    #[pyo3(signature = (t, tol = 0.0))]
    pub fn jacobian_packed_sparse_py<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        tol: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        let mut workspace = self.workspace.borrow_mut();
        let (rows, cols, values) =
            build_packed_jacobian_sparse(&self.plan, t, self.mode, &mut workspace, tol)
                .map_err(PyValueError::new_err)?;
        Ok((
            PyArray1::from_vec(py, rows),
            PyArray1::from_vec(py, cols),
            PyArray1::from_vec(py, values),
        ))
    }
}

#[pyfunction(signature = (payload))]
pub fn prepare_lindblad_problem_py(payload: &Bound<'_, PyAny>) -> PyResult<PreparedLindbladPlan> {
    parse_plan_payload(payload)
}

#[pyfunction(signature = (plan, mode = "structured"))]
pub fn create_lindblad_rhs_evaluator_py(
    plan: PyRef<'_, PreparedLindbladPlan>,
    mode: &str,
) -> PyResult<LindbladRhsEvaluator> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    Ok(LindbladRhsEvaluator {
        plan: plan.clone(),
        mode: execution_mode,
        workspace: RefCell::new(RhsWorkspace::new(&plan)),
        profiling_enabled: Cell::new(false),
        profile: RefCell::new(RhsProfileStats::default()),
    })
}

#[pyfunction(signature = (plan, packed_state, t, mode = "structured"))]
pub fn lindblad_rhs_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_state: PyReadonlyArray1<'py, f64>,
    t: f64,
    mode: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let rhs = rhs_packed(
        &plan,
        packed_state.as_slice().map_err(PyValueError::new_err)?,
        t,
        execution_mode,
    )
    .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, rhs))
}

#[pyfunction(signature = (plan, packed_vector, t, mode = "structured"))]
pub fn lindblad_jvp_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_vector: PyReadonlyArray1<'py, f64>,
    t: f64,
    mode: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let jvp = rhs_packed(
        &plan,
        packed_vector.as_slice().map_err(PyValueError::new_err)?,
        t,
        execution_mode,
    )
    .map_err(PyValueError::new_err)?;
    Ok(PyArray1::from_vec(py, jvp))
}

#[pyfunction(signature = (plan, t))]
pub fn evaluate_lindblad_hamiltonian_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    t: f64,
) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
    let matrix = plan
        .evaluate_hamiltonian(t)
        .map_err(PyValueError::new_err)?;
    let array = PyArray1::from_vec(py, matrix);
    array.reshape((plan.n_states(), plan.n_states()))
}

#[pyfunction(signature = (
    plan,
    packed_rho0,
    t0,
    t1,
    abstol,
    reltol,
    dt,
    saveat = None,
    save_start = true,
    maxiters = 100000,
    mode = "structured"
))]
pub fn solve_lindblad_dopri5_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_rho0: PyReadonlyArray1<'py, f64>,
    t0: f64,
    t1: f64,
    abstol: f64,
    reltol: f64,
    dt: f64,
    saveat: Option<PyReadonlyArray1<'py, f64>>,
    save_start: bool,
    maxiters: usize,
    mode: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let options = OdeSolverOptions {
        abstol,
        reltol,
        dt,
        saveat: saveat
            .map(|values| values.as_slice().map(|slice| slice.to_vec()))
            .transpose()
            .map_err(PyValueError::new_err)?,
        save_start,
        maxiters,
        mode: execution_mode,
    };
    let (times, states) = solve_dopri5(
        &plan,
        packed_rho0.as_slice().map_err(PyValueError::new_err)?,
        t0,
        t1,
        &options,
    )
    .map_err(PyValueError::new_err)?;
    let times_array = PyArray1::from_vec(py, times);
    let n_rows = if plan.layout.packed_len() == 0 {
        0
    } else {
        states.len() / plan.layout.packed_len()
    };
    let states_array =
        PyArray1::from_vec(py, states).reshape((n_rows, plan.layout.packed_len()))?;
    Ok((times_array, states_array))
}

#[pyfunction(signature = (
    plan,
    packed_rho0,
    t0,
    t1,
    abstol,
    reltol,
    dt,
    saveat = None,
    save_start = true,
    maxiters = 100000,
    mode = "structured"
))]
pub fn solve_lindblad_bdf_py<'py>(
    py: Python<'py>,
    plan: PyRef<'py, PreparedLindbladPlan>,
    packed_rho0: PyReadonlyArray1<'py, f64>,
    t0: f64,
    t1: f64,
    abstol: f64,
    reltol: f64,
    dt: f64,
    saveat: Option<PyReadonlyArray1<'py, f64>>,
    save_start: bool,
    maxiters: usize,
    mode: &str,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let execution_mode = ExecutionMode::from_str(mode).map_err(PyValueError::new_err)?;
    let options = BdfSolverOptions {
        abstol,
        reltol,
        dt,
        saveat: saveat
            .map(|values| values.as_slice().map(|slice| slice.to_vec()))
            .transpose()
            .map_err(PyValueError::new_err)?,
        save_start,
        maxiters,
        mode: execution_mode,
    };
    let (times, states) = solve_bdf(
        &plan,
        packed_rho0.as_slice().map_err(PyValueError::new_err)?,
        t0,
        t1,
        &options,
    )
    .map_err(PyValueError::new_err)?;
    let times_array = PyArray1::from_vec(py, times);
    let n_rows = if plan.layout.packed_len() == 0 {
        0
    } else {
        states.len() / plan.layout.packed_len()
    };
    let states_array =
        PyArray1::from_vec(py, states).reshape((n_rows, plan.layout.packed_len()))?;
    Ok((times_array, states_array))
}
