pub mod blas;
pub mod eval;
pub mod layout;
pub mod plan;
pub mod python_api;
pub mod rhs;
pub mod solver_batch;
pub mod solver_dopri5_fast;
pub mod solver_fast_common;
pub mod solver_ode;
pub mod solver_stats;
pub mod solver_tsit5_fast;

use plan::PreparedLindbladPlan;
use pyo3::prelude::*;
use python_api::{
    create_lindblad_rhs_evaluator_py, evaluate_lindblad_hamiltonian_py, lindblad_jvp_py,
    lindblad_rhs_py, prepare_lindblad_problem_py, solve_lindblad_batch_py,
    solve_lindblad_dopri5_fast_output_py, solve_lindblad_dopri5_fast_profile_py,
    solve_lindblad_dopri5_fast_py, solve_lindblad_dopri5_profile_py, solve_lindblad_dopri5_py,
    solve_lindblad_grid_batch_py, solve_lindblad_tsit5_fast_output_py,
    solve_lindblad_tsit5_fast_profile_py, solve_lindblad_tsit5_fast_py, LindbladRhsEvaluator,
};

pub fn register_python_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PreparedLindbladPlan>()?;
    m.add_class::<LindbladRhsEvaluator>()?;
    m.add_function(wrap_pyfunction!(prepare_lindblad_problem_py, m)?)?;
    m.add_function(wrap_pyfunction!(create_lindblad_rhs_evaluator_py, m)?)?;
    m.add_function(wrap_pyfunction!(lindblad_rhs_py, m)?)?;
    m.add_function(wrap_pyfunction!(lindblad_jvp_py, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_lindblad_hamiltonian_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lindblad_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lindblad_grid_batch_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lindblad_dopri5_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lindblad_dopri5_profile_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lindblad_dopri5_fast_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lindblad_dopri5_fast_profile_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lindblad_dopri5_fast_output_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lindblad_tsit5_fast_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lindblad_tsit5_fast_profile_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_lindblad_tsit5_fast_output_py, m)?)?;
    Ok(())
}
