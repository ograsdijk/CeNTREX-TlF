use crate::effective_lindblad::plan::EffectiveLindbladPlan;
use crate::lindblad::eval::RuntimeValue;
use num_complex::Complex64;

pub struct EffectiveLindbladWorkspace {
    pub parameter_values: Vec<RuntimeValue>,
    pub eval_stack: Vec<RuntimeValue>,
    pub scalar_stack: Vec<Complex64>,
    pub last_interval: usize,
    pub pchip_hints: Vec<usize>,
    pub parameter_overrides: Vec<(usize, RuntimeValue)>,
}

impl EffectiveLindbladWorkspace {
    pub fn new(plan: &EffectiveLindbladPlan) -> Self {
        Self {
            parameter_values: Vec::with_capacity(plan.parameter_graph.slot_names.len()),
            eval_stack: Vec::new(),
            scalar_stack: Vec::new(),
            last_interval: 0,
            pchip_hints: vec![0; plan.parameter_graph.pchip_tables.len()],
            parameter_overrides: Vec::new(),
        }
    }

    pub fn set_parameter_overrides(
        &mut self,
        slot_indices: &[usize],
        values: &[f64],
    ) -> Result<(), String> {
        if slot_indices.len() != values.len() {
            return Err(format!(
                "override slot count {} does not match value count {}",
                slot_indices.len(),
                values.len()
            ));
        }
        self.parameter_overrides.clear();
        for (&slot, &value) in slot_indices.iter().zip(values.iter()) {
            self.parameter_overrides
                .push((slot, RuntimeValue::Scalar(Complex64::new(value, 0.0))));
        }
        Ok(())
    }
}

fn find_interval_cached(x: f64, grid: &[f64], hint: &mut usize) -> Result<(usize, f64), String> {
    let n = grid.len();
    if n < 2 {
        return Ok((0, 0.0));
    }
    let tol = 1e-10 * (grid[n - 1] - grid[0]).abs().max(1.0);
    if x < grid[0] - tol {
        return Err(format!(
            "field coordinate {x:.6e} is below the operator grid minimum {:.6e}",
            grid[0]
        ));
    }
    if x > grid[n - 1] + tol {
        return Err(format!(
            "field coordinate {x:.6e} is above the operator grid maximum {:.6e}",
            grid[n - 1]
        ));
    }
    let x_clamped = x.clamp(grid[0], grid[n - 1]);
    if x_clamped <= grid[0] {
        *hint = 0;
        return Ok((0, 0.0));
    }
    if x_clamped >= grid[n - 1] {
        *hint = n - 2;
        return Ok((n - 2, 1.0));
    }
    let h = *hint;
    if h < n - 1 && grid[h] <= x_clamped && x_clamped <= grid[h + 1] {
        let w = (x_clamped - grid[h]) / (grid[h + 1] - grid[h]);
        return Ok((h, w));
    }
    let idx = grid
        .partition_point(|&g| g <= x_clamped)
        .saturating_sub(1)
        .min(n - 2);
    *hint = idx;
    let w = (x_clamped - grid[idx]) / (grid[idx + 1] - grid[idx]);
    Ok((idx, w))
}

pub fn rhs_effective_lindblad(
    plan: &EffectiveLindbladPlan,
    y: &[f64],
    t: f64,
    workspace: &mut EffectiveLindbladWorkspace,
    dy: &mut [f64],
) -> Result<(), String> {
    plan.parameter_graph.evaluate_with_overrides_into(
        t,
        workspace.parameter_overrides.as_slice(),
        &mut workspace.parameter_values,
        &mut workspace.eval_stack,
        &mut workspace.scalar_stack,
        &mut workspace.pchip_hints,
    )?;

    let field_val = match &workspace.parameter_values[plan.field_coordinate_slot] {
        RuntimeValue::Scalar(c) => c.re,
        _ => return Err("field_coordinate must be scalar".to_string()),
    };
    let rabi_val = plan.constant_rabi.unwrap_or_else(|| {
        match &workspace.parameter_values[plan.rabi_rate_slot] {
            RuntimeValue::Scalar(c) => c.re,
            _ => 0.0,
        }
    });
    let detuning_val = plan.constant_detuning.unwrap_or_else(|| {
        match &workspace.parameter_values[plan.detuning_slot] {
            RuntimeValue::Scalar(c) => c.re,
            _ => 0.0,
        }
    });

    let (idx, _w) =
        find_interval_cached(field_val, &plan.field_grid, &mut workspace.last_interval)?;
    let dx = field_val - plan.field_grid[idx];

    let half_rabi = 0.5 * rabi_val;

    for v in dy.iter_mut() {
        *v = 0.0;
    }
    plan.sparse_combined
        .sparse_matvec_interpolated(idx, dx, 1.0, y, dy);
    plan.sparse_opt
        .sparse_matvec_interpolated(idx, dx, half_rabi, y, dy);
    plan.sparse_det
        .sparse_matvec_interpolated(idx, dx, detuning_val, y, dy);

    Ok(())
}
