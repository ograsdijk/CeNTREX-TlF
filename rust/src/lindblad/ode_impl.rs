use crate::lindblad::eval::CompiledExpression;
use crate::lindblad::plan::PreparedLindbladPlan;
use crate::lindblad::rhs::{rhs_packed_into, ExecutionMode, RhsWorkspace};
use crate::ode::OdeRhs;
use num_complex::Complex64;

#[derive(Clone, Debug)]
pub enum LindbladStopEvent {
    RuntimeExpression {
        expression: CompiledExpression,
    },
    PopulationThreshold {
        indices: Vec<usize>,
        threshold: f64,
    },
}

pub struct LindbladRhs<'a> {
    plan: &'a PreparedLindbladPlan,
    mode: ExecutionMode,
    workspace: RhsWorkspace,
    stop_event: Option<LindbladStopEvent>,
}

impl<'a> LindbladRhs<'a> {
    pub fn new(plan: &'a PreparedLindbladPlan, mode: ExecutionMode) -> Self {
        Self {
            plan,
            mode,
            workspace: RhsWorkspace::new(plan),
            stop_event: None,
        }
    }

    pub fn with_stop_event(mut self, stop_event: Option<LindbladStopEvent>) -> Self {
        self.stop_event = stop_event;
        self
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

    pub fn new_with_overrides_and_event(
        plan: &'a PreparedLindbladPlan,
        mode: ExecutionMode,
        parameter_slot_indices: &[usize],
        parameter_values: &[Complex64],
        stop_event: Option<LindbladStopEvent>,
    ) -> Result<Self, String> {
        Ok(Self::new_with_overrides(plan, mode, parameter_slot_indices, parameter_values)?
            .with_stop_event(stop_event))
    }
}

impl OdeRhs for LindbladRhs<'_> {
    fn eval(&mut self, t: f64, y: &[f64], dy: &mut [f64]) -> Result<(), String> {
        rhs_packed_into(self.plan, y, t, self.mode, &mut self.workspace, dy)
    }

    fn dim(&self) -> usize {
        self.plan.layout.packed_len()
    }

    fn event_value(&mut self, t: f64, y: &[f64]) -> Result<Option<f64>, String> {
        let Some(event) = &self.stop_event else {
            return Ok(None);
        };
        match event {
            LindbladStopEvent::RuntimeExpression { expression } => self
                .workspace
                .evaluate_runtime_expression_real(self.plan, expression, t)
                .map(Some),
            LindbladStopEvent::PopulationThreshold { indices, threshold } => {
                let mut value = 0.0;
                for &idx in indices {
                    if idx >= self.plan.layout.n {
                        return Err(format!("population event index {idx} out of bounds"));
                    }
                    value += y[idx];
                }
                Ok(Some(value - threshold))
            }
        }
    }
}
