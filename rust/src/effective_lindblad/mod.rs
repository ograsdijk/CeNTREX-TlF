pub mod plan;
pub mod python_api;
pub mod rhs;
pub mod solver;

use plan::EffectiveLindbladPlan;
use pyo3::prelude::*;
use python_api::{
    debug_effective_lindblad_rhs_py, prepare_effective_lindblad_plan_py,
    solve_effective_lindblad_py,
};

pub fn register_python_api(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EffectiveLindbladPlan>()?;
    m.add_function(wrap_pyfunction!(prepare_effective_lindblad_plan_py, m)?)?;
    m.add_function(wrap_pyfunction!(solve_effective_lindblad_py, m)?)?;
    m.add_function(wrap_pyfunction!(debug_effective_lindblad_rhs_py, m)?)?;
    Ok(())
}
