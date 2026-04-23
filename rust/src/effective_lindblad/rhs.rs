use crate::effective_lindblad::plan::EffectiveLindbladPlan;
use crate::lindblad::eval::RuntimeValue;
use num_complex::Complex64;

pub struct EffectiveLindbladWorkspace {
    pub parameter_values: Vec<RuntimeValue>,
    pub eval_stack: Vec<RuntimeValue>,
    pub scalar_stack: Vec<Complex64>,
    pub l_scratch: Vec<f64>,
    pub last_interval: usize,
}

impl EffectiveLindbladWorkspace {
    pub fn new(plan: &EffectiveLindbladPlan) -> Self {
        let mat_size = plan.real_dim * plan.real_dim;
        Self {
            parameter_values: Vec::with_capacity(plan.parameter_graph.slot_names.len()),
            eval_stack: Vec::new(),
            scalar_stack: Vec::new(),
            l_scratch: vec![0.0; mat_size],
            last_interval: 0,
        }
    }
}

fn find_interval_cached(x: f64, grid: &[f64], hint: &mut usize) -> (usize, f64) {
    let n = grid.len();
    if n < 2 {
        return (0, 0.0);
    }
    if x <= grid[0] {
        *hint = 0;
        return (0, 0.0);
    }
    if x >= grid[n - 1] {
        *hint = n - 2;
        return (n - 2, 1.0);
    }
    let h = *hint;
    if h < n - 1 && grid[h] <= x && x <= grid[h + 1] {
        let w = (x - grid[h]) / (grid[h + 1] - grid[h]);
        return (h, w);
    }
    let idx = grid
        .partition_point(|&g| g <= x)
        .saturating_sub(1)
        .min(n - 2);
    *hint = idx;
    let w = (x - grid[idx]) / (grid[idx + 1] - grid[idx]);
    (idx, w)
}

pub fn rhs_effective_lindblad(
    plan: &EffectiveLindbladPlan,
    y: &[f64],
    t: f64,
    workspace: &mut EffectiveLindbladWorkspace,
    dy: &mut [f64],
) -> Result<(), String> {
    let dim = plan.real_dim;
    let mat_size = dim * dim;

    plan.parameter_graph.evaluate_into(
        t,
        &mut workspace.parameter_values,
        &mut workspace.eval_stack,
        &mut workspace.scalar_stack,
    )?;

    let field_val = match &workspace.parameter_values[plan.field_coordinate_slot] {
        RuntimeValue::Scalar(c) => c.re,
        _ => return Err("field_coordinate must be scalar".to_string()),
    };
    let rabi_val = match &workspace.parameter_values[plan.rabi_rate_slot] {
        RuntimeValue::Scalar(c) => c.re,
        _ => return Err("rabi_rate must be scalar".to_string()),
    };
    let detuning_val = match &workspace.parameter_values[plan.detuning_slot] {
        RuntimeValue::Scalar(c) => c.re,
        _ => return Err("detuning must be scalar".to_string()),
    };

    let (idx, w) = find_interval_cached(field_val, &plan.field_grid, &mut workspace.last_interval);

    let l_scratch = &mut workspace.l_scratch;
    let base_combined = idx * mat_size;
    let base_opt = idx * mat_size;
    let base_det = idx * mat_size;
    let half_rabi = 0.5 * rabi_val;

    if w == 0.0 && idx == 0 {
        for k in 0..mat_size {
            l_scratch[k] = plan.l_combined[base_combined + k]
                + half_rabi * plan.l_opt[base_opt + k]
                + detuning_val * plan.l_det[base_det + k];
        }
    } else {
        let diff_base = idx.min(plan.n_grid - 2) * mat_size;
        for k in 0..mat_size {
            let lc = plan.l_combined[base_combined + k] + w * plan.dl_combined[diff_base + k];
            let lo = plan.l_opt[base_opt + k] + w * plan.dl_opt[diff_base + k];
            let ld = plan.l_det[base_det + k] + w * plan.dl_det[diff_base + k];
            l_scratch[k] = lc + half_rabi * lo + detuning_val * ld;
        }
    }

    for i in 0..dim {
        let mut sum = 0.0;
        let row_base = i * dim;
        for j in 0..dim {
            sum += l_scratch[row_base + j] * y[j];
        }
        dy[i] = sum;
    }

    Ok(())
}
