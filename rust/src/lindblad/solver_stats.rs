#[derive(Clone, Debug, Default)]
pub struct SolveStats {
    pub solver: String,
    pub rhs_calls: u64,
    pub jacobian_calls: u64,
    pub function_evaluations: u64,
    pub accepted_steps: u64,
    pub rejected_steps: u64,
    pub internal_steps: u64,
    pub saved_points: u64,
    pub setup_seconds: f64,
    pub integration_seconds: f64,
    pub interpolation_seconds: f64,
    pub total_seconds: f64,
    pub rhs_seconds: f64,
    pub jacobian_seconds: f64,
}

impl SolveStats {
    pub fn non_rhs_seconds(&self) -> f64 {
        (self.total_seconds - self.rhs_seconds - self.jacobian_seconds).max(0.0)
    }

    pub fn average_rhs_seconds(&self) -> f64 {
        if self.rhs_calls == 0 {
            0.0
        } else {
            self.rhs_seconds / self.rhs_calls as f64
        }
    }

    pub fn average_jacobian_seconds(&self) -> f64 {
        if self.jacobian_calls == 0 {
            0.0
        } else {
            self.jacobian_seconds / self.jacobian_calls as f64
        }
    }
}
