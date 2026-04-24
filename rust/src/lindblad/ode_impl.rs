use crate::lindblad::plan::PreparedLindbladPlan;
use crate::lindblad::rhs::{rhs_packed_into, ExecutionMode, RhsWorkspace};
use crate::ode::OdeRhs;
use num_complex::Complex64;

pub struct LindbladRhs<'a> {
    plan: &'a PreparedLindbladPlan,
    mode: ExecutionMode,
    workspace: RhsWorkspace,
}

impl<'a> LindbladRhs<'a> {
    pub fn new(plan: &'a PreparedLindbladPlan, mode: ExecutionMode) -> Self {
        Self {
            plan,
            mode,
            workspace: RhsWorkspace::new(plan),
        }
    }

    pub fn new_with_overrides(
        plan: &'a PreparedLindbladPlan,
        mode: ExecutionMode,
        parameter_slot_indices: &[usize],
        parameter_values: &[Complex64],
    ) -> Result<Self, String> {
        let mut rhs = Self::new(plan, mode);
        rhs.workspace
            .set_scalar_parameter_overrides(parameter_slot_indices, parameter_values)?;
        Ok(rhs)
    }
}

impl OdeRhs for LindbladRhs<'_> {
    fn eval(&mut self, t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), String> {
        rhs_packed_into(self.plan, y, t, self.mode, &mut self.workspace, dy)
    }

    fn dim(&self) -> usize {
        self.plan.layout.packed_len()
    }
}
