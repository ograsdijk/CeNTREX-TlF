use crate::constants::ED_XTB;
use num_complex::Complex64;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub enum RuntimeValue {
    Scalar(Complex64),
    Tuple(Vec<Complex64>),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum InstructionOp {
    Const = 1,
    Slot = 2,
    Temp = 3,
    Time = 4,
    Add = 5,
    Sub = 6,
    Mul = 7,
    Div = 8,
    Pow = 9,
    Neg = 10,
    Conj = 11,
    BuiltinFunc = 12,
    HelperFunc = 13,
    Gt = 14,
    Ge = 15,
    Lt = 16,
    Le = 17,
    Eq = 18,
    Ne = 19,
}

impl InstructionOp {
    pub fn from_i64(value: i64) -> Result<Self, String> {
        match value {
            1 => Ok(Self::Const),
            2 => Ok(Self::Slot),
            3 => Ok(Self::Temp),
            4 => Ok(Self::Time),
            5 => Ok(Self::Add),
            6 => Ok(Self::Sub),
            7 => Ok(Self::Mul),
            8 => Ok(Self::Div),
            9 => Ok(Self::Pow),
            10 => Ok(Self::Neg),
            11 => Ok(Self::Conj),
            12 => Ok(Self::BuiltinFunc),
            13 => Ok(Self::HelperFunc),
            14 => Ok(Self::Gt),
            15 => Ok(Self::Ge),
            16 => Ok(Self::Lt),
            17 => Ok(Self::Le),
            18 => Ok(Self::Eq),
            19 => Ok(Self::Ne),
            _ => Err(format!("unknown instruction op {}", value)),
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum BuiltinFunctionId {
    Sin = 1,
    Cos = 2,
    Tan = 3,
    Exp = 4,
    Abs = 5,
    Real = 6,
    Imag = 7,
}

impl BuiltinFunctionId {
    fn from_i64(value: i64) -> Result<Self, String> {
        match value {
            1 => Ok(Self::Sin),
            2 => Ok(Self::Cos),
            3 => Ok(Self::Tan),
            4 => Ok(Self::Exp),
            5 => Ok(Self::Abs),
            6 => Ok(Self::Real),
            7 => Ok(Self::Imag),
            _ => Err(format!("unknown builtin function id {}", value)),
        }
    }
}

fn as_scalar(value: &RuntimeValue) -> Result<Complex64, String> {
    match value {
        RuntimeValue::Scalar(value) => Ok(*value),
        RuntimeValue::Tuple(_) => Err("sequence value used where scalar was required".to_string()),
    }
}

fn as_real(value: &RuntimeValue) -> Result<f64, String> {
    let scalar = as_scalar(value)?;
    if scalar.im.abs() > 1e-12 {
        return Err("complex value used where real scalar was required".to_string());
    }
    Ok(scalar.re)
}

fn binary_elementwise<F>(
    left: &RuntimeValue,
    right: &RuntimeValue,
    op: F,
) -> Result<RuntimeValue, String>
where
    F: Fn(Complex64, Complex64) -> Complex64,
{
    match (left, right) {
        (RuntimeValue::Scalar(left), RuntimeValue::Scalar(right)) => {
            Ok(RuntimeValue::Scalar(op(*left, *right)))
        }
        (RuntimeValue::Tuple(left), RuntimeValue::Tuple(right)) => {
            if left.len() != right.len() {
                return Err("tuple lengths do not match for elementwise operation".to_string());
            }
            Ok(RuntimeValue::Tuple(
                left.iter()
                    .zip(right.iter())
                    .map(|(l, r)| op(*l, *r))
                    .collect(),
            ))
        }
        (RuntimeValue::Tuple(left), RuntimeValue::Scalar(right)) => Ok(RuntimeValue::Tuple(
            left.iter().map(|item| op(*item, *right)).collect(),
        )),
        (RuntimeValue::Scalar(left), RuntimeValue::Tuple(right)) => Ok(RuntimeValue::Tuple(
            right.iter().map(|item| op(*left, *item)).collect(),
        )),
    }
}

fn apply_builtin_scalar(function_id: i64, value: Complex64) -> Result<Complex64, String> {
    let builtin = BuiltinFunctionId::from_i64(function_id)?;
    Ok(match builtin {
        BuiltinFunctionId::Sin => value.sin(),
        BuiltinFunctionId::Cos => value.cos(),
        BuiltinFunctionId::Tan => value.tan(),
        BuiltinFunctionId::Exp => value.exp(),
        BuiltinFunctionId::Abs => Complex64::new(value.norm(), 0.0),
        BuiltinFunctionId::Real => Complex64::new(value.re, 0.0),
        BuiltinFunctionId::Imag => Complex64::new(value.im, 0.0),
    })
}

fn apply_builtin(function_id: i64, args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    Ok(RuntimeValue::Scalar(apply_builtin_scalar(
        function_id,
        as_scalar(&args[0])?,
    )?))
}

fn gaussian_2d(
    x: f64,
    y: f64,
    amplitude: f64,
    mean_x: f64,
    mean_y: f64,
    sigma_x: f64,
    sigma_y: f64,
) -> f64 {
    let dx = x - mean_x;
    let dy = y - mean_y;
    amplitude
        * (-((dx * dx) / (2.0 * sigma_x * sigma_x) + (dy * dy) / (2.0 * sigma_y * sigma_y))).exp()
}

fn gaussian_2d_rotated(
    x: f64,
    y: f64,
    amplitude: f64,
    mean_x: f64,
    mean_y: f64,
    sigma_x: f64,
    sigma_y: f64,
    theta: f64,
) -> f64 {
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();
    let a = cos_theta * cos_theta / (2.0 * sigma_x * sigma_x)
        + sin_theta * sin_theta / (2.0 * sigma_y * sigma_y);
    let b = (2.0 * theta).sin() / (2.0 * sigma_x * sigma_x)
        - (2.0 * theta).sin() / (2.0 * sigma_y * sigma_y);
    let c = sin_theta * sin_theta / (2.0 * sigma_x * sigma_x)
        + cos_theta * cos_theta / (2.0 * sigma_y * sigma_y);
    let dx = x - mean_x;
    let dy = y - mean_y;
    amplitude * (-(a * dx * dx + b * dx * dy + c * dy * dy)).exp()
}

fn phase_modulation(t: f64, beta: f64, omega: f64) -> Complex64 {
    let phi = beta * (omega * t).sin();
    Complex64::new(phi.cos(), phi.sin())
}

fn square_wave(t: f64, omega: f64, phase: f64) -> f64 {
    0.5 * (1.0
        + if (omega * t + phase).sin() >= 0.0 {
            1.0
        } else {
            -1.0
        })
}

fn resonant_polarization_modulation(t: f64, gamma: f64, omega: f64) -> Complex64 {
    let theta = 0.5 * gamma * (omega * t).sin();
    let a = 0.5 * (theta.cos() + theta.sin());
    Complex64::new(a, a)
}

/// Sawtooth ramp from 0 to 1 at frequency `omega` [rad/s], phase `phase` [rad].
///
/// No `- PI` here, even though the Julia original reads
/// `0.5*(1 + sawtoothwave(omega*t + phase - pi))`. Julia's
/// `Waveforms.sawtoothwave` is zero-centred (`rem2pi(x, RoundNearest)/pi`,
/// range (-pi, pi]), so its `- pi` supplies a half-period shift that the
/// floor-based `rem_euclid(1.0)` used here already accounts for. Keeping both
/// applied the shift twice and put phase=0 halfway up the ramp. Do not
/// "restore" it. Must stay in lockstep with the Python
/// `helper_functions.sawtooth_wave`.
fn sawtooth_wave(t: f64, omega: f64, phase: f64) -> f64 {
    ((omega * t + phase) / (2.0 * PI)).rem_euclid(1.0)
}

fn variable_on_off(t: f64, ton: f64, toff: f64, phase: f64) -> f64 {
    let period = ton + toff;
    let duty = ton / period;
    let frac = (2.0 * PI * t / period + phase).rem_euclid(2.0 * PI) / (2.0 * PI);
    if frac < duty {
        1.0
    } else {
        0.0
    }
}

fn variable_on_off_duty(t: f64, duty: f64, inv_period: f64, phase: f64) -> f64 {
    let mut frac = (t * inv_period + phase / (2.0 * PI)).rem_euclid(1.0);
    if frac <= 0.0 {
        frac += 1.0;
    }
    if frac < duty {
        1.0
    } else {
        0.0
    }
}

fn multipass_2d_intensity(
    x: f64,
    y: f64,
    amplitudes: &[Complex64],
    xlocs: &[Complex64],
    ylocs: &[Complex64],
    sigma_x: f64,
    sigma_y: f64,
) -> Result<f64, String> {
    if amplitudes.len() != xlocs.len() || amplitudes.len() != ylocs.len() {
        return Err("multipass helper tuple lengths do not match".to_string());
    }
    let mut intensity = 0.0;
    for idx in 0..amplitudes.len() {
        intensity += gaussian_2d(
            x,
            y,
            as_real(&RuntimeValue::Scalar(amplitudes[idx]))?,
            as_real(&RuntimeValue::Scalar(xlocs[idx]))?,
            as_real(&RuntimeValue::Scalar(ylocs[idx]))?,
            sigma_x,
            sigma_y,
        );
    }
    Ok(intensity)
}

fn rabi_from_intensity(intensity: f64, coupling: f64, dipole_moment: f64) -> f64 {
    let hbar = 1.0545718176461565e-34;
    let c = 299792458.0;
    let eps0 = 8.8541878128e-12;
    let electric_field = (intensity * 2.0 / (c * eps0)).sqrt();
    electric_field * coupling * dipole_moment / hbar
}

fn gaussian_1d(x: f64, center: f64, sigma: f64) -> f64 {
    let dx = x - center;
    (-dx * dx / (2.0 * sigma * sigma)).exp()
}

fn pchip_sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Node derivative of SciPy's `PchipInterpolator` at grid point `k`, computed
/// from the neighbouring secants only, so evaluating one interval needs no
/// allocation. Mirrors `scipy.interpolate._cubic.PchipInterpolator._find_derivatives`,
/// including the three-point shape-preserving end slopes (`_edge_case`) and the
/// linear two-point case. Must stay in lockstep with the Python helper, which
/// calls SciPy directly.
fn pchip_node_slope(g: &[Complex64], v: &[Complex64], k: usize) -> f64 {
    let n = g.len();
    let h = |i: usize| g[i + 1].re - g[i].re;
    let m = |i: usize| (v[i + 1].re - v[i].re) / h(i);
    if n == 2 {
        return m(0);
    }
    let edge = |h0: f64, h1: f64, m0: f64, m1: f64| {
        let d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
        if pchip_sign(d) != pchip_sign(m0) {
            0.0
        } else if pchip_sign(m0) != pchip_sign(m1) && d.abs() > 3.0 * m0.abs() {
            3.0 * m0
        } else {
            d
        }
    };
    if k == 0 {
        return edge(h(0), h(1), m(0), m(1));
    }
    if k == n - 1 {
        return edge(h(n - 2), h(n - 3), m(n - 2), m(n - 3));
    }
    let (m_left, m_right) = (m(k - 1), m(k));
    if pchip_sign(m_left) != pchip_sign(m_right) || m_left == 0.0 || m_right == 0.0 {
        return 0.0;
    }
    let (h_left, h_right) = (h(k - 1), h(k));
    let w1 = 2.0 * h_right + h_left;
    let w2 = h_right + 2.0 * h_left;
    (w1 + w2) / (w1 / m_left + w2 / m_right)
}

/// PCHIP interpolation matching `scipy.interpolate.PchipInterpolator(...,
/// extrapolate=True)`: outside the grid the end cubic is extended.
fn pchip_interp(x: f64, grid: &[Complex64], values: &[Complex64]) -> Result<f64, String> {
    if grid.len() != values.len() {
        return Err("pchip_interp grid and values must have the same length".to_string());
    }
    let n = grid.len();
    if n < 2 {
        if n == 1 {
            return as_real(&RuntimeValue::Scalar(values[0]));
        }
        return Err("pchip_interp grid must contain at least one point".to_string());
    }
    if grid.windows(2).any(|pair| !(pair[1].re > pair[0].re)) {
        return Err("pchip_interp grid must be strictly increasing".to_string());
    }
    let idx = grid
        .partition_point(|g| g.re <= x)
        .saturating_sub(1)
        .min(n - 2);
    let h = grid[idx + 1].re - grid[idx].re;
    let d_left = pchip_node_slope(grid, values, idx);
    let d_right = pchip_node_slope(grid, values, idx + 1);
    let t = (x - grid[idx].re) / h;
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    Ok(h00 * values[idx].re + h10 * h * d_left + h01 * values[idx + 1].re + h11 * h * d_right)
}

fn helper_args_to_tuple(arg: &RuntimeValue) -> Result<&[Complex64], String> {
    match arg {
        RuntimeValue::Tuple(values) => Ok(values.as_slice()),
        RuntimeValue::Scalar(_) => Err("helper expected tuple argument".to_string()),
    }
}

fn linear_interp(x: f64, grid: &[Complex64], values: &[Complex64]) -> Result<f64, String> {
    if grid.len() != values.len() {
        return Err("linear_interp grid and values must have the same length".to_string());
    }
    if grid.is_empty() {
        return Err("linear_interp grid must contain at least one point".to_string());
    }
    if grid.len() == 1 {
        return as_real(&RuntimeValue::Scalar(values[0]));
    }
    let first = as_real(&RuntimeValue::Scalar(grid[0]))?;
    if x <= first {
        return as_real(&RuntimeValue::Scalar(values[0]));
    }
    for idx in 0..(grid.len() - 1) {
        let left = as_real(&RuntimeValue::Scalar(grid[idx]))?;
        let right = as_real(&RuntimeValue::Scalar(grid[idx + 1]))?;
        if right <= left {
            return Err("linear_interp grid must be strictly increasing".to_string());
        }
        if x <= right {
            let y_left = as_real(&RuntimeValue::Scalar(values[idx]))?;
            let y_right = as_real(&RuntimeValue::Scalar(values[idx + 1]))?;
            let frac = (x - left) / (right - left);
            return Ok(y_left + frac * (y_right - y_left));
        }
    }
    as_real(&RuntimeValue::Scalar(values[values.len() - 1]))
}

fn apply_helper(function_id: i64, args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    match function_id {
        8 => Ok(RuntimeValue::Scalar(Complex64::new(
            multipass_2d_intensity(
                as_real(&args[0])?,
                as_real(&args[1])?,
                helper_args_to_tuple(&args[2])?,
                helper_args_to_tuple(&args[3])?,
                helper_args_to_tuple(&args[4])?,
                as_real(&args[5])?,
                as_real(&args[6])?,
            )?,
            0.0,
        ))),
        10 => {
            let intensity = multipass_2d_intensity(
                as_real(&args[0])?,
                as_real(&args[1])?,
                helper_args_to_tuple(&args[2])?,
                helper_args_to_tuple(&args[3])?,
                helper_args_to_tuple(&args[4])?,
                as_real(&args[5])?,
                as_real(&args[6])?,
            )?;
            Ok(RuntimeValue::Scalar(Complex64::new(
                rabi_from_intensity(
                    intensity,
                    as_real(&args[7])?,
                    if args.len() > 8 {
                        as_real(&args[8])?
                    } else {
                        ED_XTB
                    },
                ),
                0.0,
            )))
        }
        14 => Ok(RuntimeValue::Scalar(Complex64::new(
            linear_interp(
                as_real(&args[0])?,
                helper_args_to_tuple(&args[1])?,
                helper_args_to_tuple(&args[2])?,
            )?,
            0.0,
        ))),
        16 => Ok(RuntimeValue::Scalar(Complex64::new(
            pchip_interp(
                as_real(&args[0])?,
                helper_args_to_tuple(&args[1])?,
                helper_args_to_tuple(&args[2])?,
            )?,
            0.0,
        ))),
        _ => {
            let scalar_args: Result<Vec<Complex64>, _> = args.iter().map(as_scalar).collect();
            Ok(RuntimeValue::Scalar(apply_helper_scalar(
                function_id,
                &scalar_args?,
            )?))
        }
    }
}

fn compare(
    left: &RuntimeValue,
    right: &RuntimeValue,
    op: InstructionOp,
) -> Result<RuntimeValue, String> {
    let lhs = as_real(left)?;
    let rhs = as_real(right)?;
    let result = match op {
        InstructionOp::Gt => lhs > rhs,
        InstructionOp::Ge => lhs >= rhs,
        InstructionOp::Lt => lhs < rhs,
        InstructionOp::Le => lhs <= rhs,
        InstructionOp::Eq => lhs == rhs,
        InstructionOp::Ne => lhs != rhs,
        _ => return Err(format!("unsupported comparison op {:?}", op)),
    };
    Ok(RuntimeValue::Scalar(Complex64::new(
        if result { 1.0 } else { 0.0 },
        0.0,
    )))
}

#[derive(Clone, Debug)]
pub struct Instruction {
    pub op: InstructionOp,
    pub index: usize,
    pub argc: usize,
    pub function: i64,
    pub value: Complex64,
}

#[derive(Clone, Debug)]
pub struct CompiledExpression {
    pub instructions: Vec<Instruction>,
    pub scalar_only: bool,
    pub output_is_tuple: bool,
}

/// Accepted argument counts per helper id; must match `HelperFunctionId` and
/// the Python helper signatures in `centrex_tlf/lindblad/helper_functions.py`.
fn helper_arity_ok(function: i64, argc: usize) -> bool {
    match function {
        1 | 8 => argc == 7,
        2 => argc == 8,
        3 | 4 | 5 | 6 | 13 | 14 | 15 | 16 => argc == 3,
        7 | 12 => argc == 4,
        9 => argc == 2 || argc == 3,
        10 | 11 => argc == 8 || argc == 9,
        _ => false,
    }
}

impl CompiledExpression {
    /// Check stack discipline, builtin/helper arity and slot/temp indices so a
    /// malformed payload raises when the plan is parsed instead of panicking
    /// inside the RHS.
    pub fn validate(&self, n_slots: usize, n_temps: usize) -> Result<(), String> {
        let mut depth = 0usize;
        for (position, ins) in self.instructions.iter().enumerate() {
            let (pops, label) = match ins.op {
                InstructionOp::Const | InstructionOp::Time => (0, "operand"),
                InstructionOp::Slot => {
                    if ins.index >= n_slots {
                        return Err(format!(
                            "instruction {position} reads slot {} but only {n_slots} slots exist",
                            ins.index
                        ));
                    }
                    (0, "SLOT")
                }
                InstructionOp::Temp => {
                    if ins.index >= n_temps {
                        return Err(format!(
                            "instruction {position} reads temp {} but only {n_temps} temps are available",
                            ins.index
                        ));
                    }
                    (0, "TEMP")
                }
                InstructionOp::Neg | InstructionOp::Conj => (1, "unary op"),
                InstructionOp::BuiltinFunc => {
                    BuiltinFunctionId::from_i64(ins.function)?;
                    if ins.argc != 1 {
                        return Err(format!(
                            "builtin function {} takes 1 argument, got {}",
                            ins.function, ins.argc
                        ));
                    }
                    (1, "BUILTIN_FUNC")
                }
                InstructionOp::HelperFunc => {
                    if !helper_arity_ok(ins.function, ins.argc) {
                        return Err(format!(
                            "helper function {} does not accept {} arguments",
                            ins.function, ins.argc
                        ));
                    }
                    (ins.argc, "HELPER_FUNC")
                }
                _ => (2, "binary op"),
            };
            depth = depth
                .checked_sub(pops)
                .ok_or_else(|| format!("stack underflow on {label} at instruction {position}"))?;
            depth += 1;
        }
        if depth != 1 {
            return Err(format!(
                "expression leaves {depth} values on the stack instead of 1"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct ScalarDagNode {
    instruction: Instruction,
    args: Vec<usize>,
}

/// Immutable shared scalar coefficient expressions. Evaluation scratch belongs
/// to the caller so cloned plans can be used by independent batch workers.
#[derive(Clone, Debug)]
pub struct ScalarCoefficientDag {
    nodes: Vec<ScalarDagNode>,
    roots: Vec<usize>,
}

impl ScalarCoefficientDag {
    pub fn prepare<'a>(expressions: impl Iterator<Item = &'a CompiledExpression>) -> Option<Self> {
        let mut nodes = Vec::new();
        let mut roots = Vec::new();
        let mut intern = std::collections::HashMap::new();
        let mut original_count = 0;
        for expression in expressions {
            if !expression.scalar_only || expression.output_is_tuple {
                return None;
            }
            let mut stack: Vec<usize> = Vec::new();
            for ins in &expression.instructions {
                original_count += 1;
                let argc = match ins.op {
                    InstructionOp::Const | InstructionOp::Slot | InstructionOp::Time => 0,
                    // Decomposed coefficients do not normally reference temps.
                    // Keep the existing evaluator for any expression that does.
                    InstructionOp::Temp => return None,
                    InstructionOp::Neg | InstructionOp::Conj => 1,
                    InstructionOp::BuiltinFunc | InstructionOp::HelperFunc => ins.argc,
                    _ => 2,
                };
                let start = stack.len().checked_sub(argc)?;
                let args: Vec<usize> = stack.drain(start..).collect();
                let key = (
                    ins.op as u8,
                    ins.index,
                    ins.argc,
                    ins.function,
                    ins.value.re.to_bits(),
                    ins.value.im.to_bits(),
                    args.clone(),
                );
                let idx = *intern.entry(key).or_insert_with(|| {
                    let idx = nodes.len();
                    nodes.push(ScalarDagNode {
                        instruction: ins.clone(),
                        args,
                    });
                    idx
                });
                stack.push(idx);
            }
            if stack.len() != 1 {
                return None;
            }
            roots.push(stack[0]);
        }
        // No shared work: retain the cheaper existing expression evaluator.
        if nodes.len() >= original_count {
            return None;
        }
        Some(Self { nodes, roots })
    }

    pub fn evaluate_into(
        &self,
        slots: &[RuntimeValue],
        t: f64,
        values: &mut Vec<RuntimeValue>,
        args: &mut Vec<Complex64>,
    ) -> Result<(), String> {
        values.clear();
        for node in &self.nodes {
            let ins = &node.instruction;
            let arg = |i: usize| as_scalar(&values[node.args[i]]);
            let value = match ins.op {
                InstructionOp::Const => ins.value,
                InstructionOp::Slot => as_scalar(&slots[ins.index])?,
                InstructionOp::Time => Complex64::new(t, 0.0),
                InstructionOp::Add => arg(0)? + arg(1)?,
                InstructionOp::Sub => arg(0)? - arg(1)?,
                InstructionOp::Mul => arg(0)? * arg(1)?,
                InstructionOp::Div => arg(0)? / arg(1)?,
                InstructionOp::Pow => arg(0)?.powc(arg(1)?),
                InstructionOp::Neg => -arg(0)?,
                InstructionOp::Conj => arg(0)?.conj(),
                InstructionOp::BuiltinFunc => apply_builtin_scalar(ins.function, arg(0)?)?,
                InstructionOp::HelperFunc => {
                    args.clear();
                    for &idx in &node.args {
                        args.push(as_scalar(&values[idx])?);
                    }
                    apply_helper_scalar(ins.function, args)?
                }
                InstructionOp::Gt
                | InstructionOp::Ge
                | InstructionOp::Lt
                | InstructionOp::Le
                | InstructionOp::Eq
                | InstructionOp::Ne => {
                    let a = arg(0)?.re;
                    let b = arg(1)?.re;
                    let result = match ins.op {
                        InstructionOp::Gt => a > b,
                        InstructionOp::Ge => a >= b,
                        InstructionOp::Lt => a < b,
                        InstructionOp::Le => a <= b,
                        InstructionOp::Eq => a == b,
                        InstructionOp::Ne => a != b,
                        _ => unreachable!(),
                    };
                    Complex64::new(if result { 1.0 } else { 0.0 }, 0.0)
                }
                InstructionOp::Temp => unreachable!("temps use the original evaluator"),
            };
            values.push(RuntimeValue::Scalar(value));
        }
        Ok(())
    }

    pub fn coefficient(&self, index: usize, values: &[RuntimeValue]) -> Complex64 {
        match values[self.roots[index]] {
            RuntimeValue::Scalar(value) => value,
            RuntimeValue::Tuple(_) => unreachable!("scalar DAG values"),
        }
    }
}

#[cfg(test)]
mod coefficient_dag_tests {
    use super::*;

    fn instruction(op: InstructionOp, index: usize, argc: usize, function: i64) -> Instruction {
        Instruction {
            op,
            index,
            argc,
            function,
            value: Complex64::new(0.0, 1.0),
        }
    }

    #[test]
    fn shared_complex_coefficients_follow_time_and_slot_updates() {
        let prefix = vec![
            instruction(InstructionOp::Slot, 0, 0, 0),
            instruction(InstructionOp::Time, 0, 0, 0),
            instruction(InstructionOp::Add, 0, 0, 0),
            instruction(InstructionOp::BuiltinFunc, 0, 1, 1),
        ];
        let mut a = prefix.clone();
        a.extend([
            instruction(InstructionOp::Slot, 1, 0, 0),
            instruction(InstructionOp::Mul, 0, 0, 0),
        ]);
        let mut b = prefix;
        b.extend([
            instruction(InstructionOp::Conj, 0, 0, 0),
            instruction(InstructionOp::Const, 0, 0, 0),
            instruction(InstructionOp::Mul, 0, 0, 0),
        ]);
        let expressions = [a, b].map(|instructions| CompiledExpression {
            instructions,
            scalar_only: true,
            output_is_tuple: false,
        });
        let dag = ScalarCoefficientDag::prepare(expressions.iter()).unwrap();
        assert!(dag.nodes.len() < expressions.iter().map(|e| e.instructions.len()).sum());
        let mut values = Vec::new();
        let mut scratch = Vec::new();
        for x in [0.0, 0.3, 1.7] {
            for t in [0.0, 0.23, 1.2] {
                let slots = [
                    RuntimeValue::Scalar(Complex64::new(x, 0.2)),
                    RuntimeValue::Scalar(Complex64::new(0.4, -x)),
                ];
                dag.evaluate_into(&slots, t, &mut values, &mut scratch)
                    .unwrap();
                for (i, e) in expressions.iter().enumerate() {
                    let expected =
                        eval_scalar_expression_into(e, &slots, t, &[], &mut scratch).unwrap();
                    assert!((dag.coefficient(i, &values) - expected).norm() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn shared_helpers_match_existing_evaluator() {
        let expression = CompiledExpression {
            scalar_only: true,
            output_is_tuple: false,
            instructions: vec![
                instruction(InstructionOp::Time, 0, 0, 0),
                instruction(InstructionOp::Slot, 0, 0, 0),
                instruction(InstructionOp::Slot, 1, 0, 0),
                instruction(InstructionOp::HelperFunc, 0, 3, 3),
            ],
        };
        let expressions = [expression.clone(), expression];
        let dag = ScalarCoefficientDag::prepare(expressions.iter()).unwrap();
        let slots = [
            RuntimeValue::Scalar(Complex64::new(0.7, 0.0)),
            RuntimeValue::Scalar(Complex64::new(2.1, 0.0)),
        ];
        let mut values = Vec::new();
        let mut scratch = Vec::new();
        for t in [0.0, 0.2, 1.5] {
            dag.evaluate_into(&slots, t, &mut values, &mut scratch)
                .unwrap();
            let reference =
                eval_scalar_expression_into(&expressions[0], &slots, t, &[], &mut scratch).unwrap();
            assert_eq!(dag.coefficient(0, &values), reference);
            assert_eq!(dag.coefficient(1, &values), reference);
        }
    }

    #[test]
    fn unsupported_or_unshared_expressions_keep_existing_evaluator() {
        let mut expression = CompiledExpression {
            scalar_only: true,
            output_is_tuple: false,
            instructions: vec![instruction(InstructionOp::Slot, 0, 0, 0)],
        };
        assert!(ScalarCoefficientDag::prepare([&expression].into_iter()).is_none());
        expression.scalar_only = false;
        assert!(ScalarCoefficientDag::prepare([&expression, &expression].into_iter()).is_none());
        expression.scalar_only = true;
        expression.instructions[0].op = InstructionOp::Temp;
        assert!(ScalarCoefficientDag::prepare([&expression, &expression].into_iter()).is_none());
    }
}

pub fn eval_expression_into(
    expression: &CompiledExpression,
    slots: &[RuntimeValue],
    t: f64,
    temps: &[RuntimeValue],
    stack: &mut Vec<RuntimeValue>,
) -> Result<RuntimeValue, String> {
    stack.clear();
    for instruction in &expression.instructions {
        match instruction.op {
            InstructionOp::Const => stack.push(RuntimeValue::Scalar(instruction.value)),
            InstructionOp::Slot => stack.push(slots[instruction.index].clone()),
            InstructionOp::Temp => stack.push(temps[instruction.index].clone()),
            InstructionOp::Time => stack.push(RuntimeValue::Scalar(Complex64::new(t, 0.0))),
            InstructionOp::Add => {
                let right = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on ADD".to_string())?;
                let left = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on ADD".to_string())?;
                stack.push(binary_elementwise(&left, &right, |a, b| a + b)?);
            }
            InstructionOp::Sub => {
                let right = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on SUB".to_string())?;
                let left = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on SUB".to_string())?;
                stack.push(binary_elementwise(&left, &right, |a, b| a - b)?);
            }
            InstructionOp::Mul => {
                let right = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on MUL".to_string())?;
                let left = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on MUL".to_string())?;
                stack.push(binary_elementwise(&left, &right, |a, b| a * b)?);
            }
            InstructionOp::Div => {
                let right = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on DIV".to_string())?;
                let left = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on DIV".to_string())?;
                stack.push(binary_elementwise(&left, &right, |a, b| a / b)?);
            }
            InstructionOp::Pow => {
                let right = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on POW".to_string())?;
                let left = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on POW".to_string())?;
                stack.push(binary_elementwise(&left, &right, |a, b| a.powc(b))?);
            }
            InstructionOp::Neg => match stack
                .pop()
                .ok_or_else(|| "stack underflow on NEG".to_string())?
            {
                RuntimeValue::Scalar(value) => stack.push(RuntimeValue::Scalar(-value)),
                RuntimeValue::Tuple(values) => stack.push(RuntimeValue::Tuple(
                    values.into_iter().map(|value| -value).collect(),
                )),
            },
            InstructionOp::Conj => match stack
                .pop()
                .ok_or_else(|| "stack underflow on CONJ".to_string())?
            {
                RuntimeValue::Scalar(value) => stack.push(RuntimeValue::Scalar(value.conj())),
                RuntimeValue::Tuple(values) => stack.push(RuntimeValue::Tuple(
                    values.into_iter().map(|value| value.conj()).collect(),
                )),
            },
            InstructionOp::BuiltinFunc => {
                let argc = instruction.argc;
                let start = stack
                    .len()
                    .checked_sub(argc)
                    .ok_or_else(|| "stack underflow on BUILTIN_FUNC".to_string())?;
                let result = apply_builtin(instruction.function, &stack[start..])?;
                stack.truncate(start);
                stack.push(result);
            }
            InstructionOp::HelperFunc => {
                let argc = instruction.argc;
                let start = stack
                    .len()
                    .checked_sub(argc)
                    .ok_or_else(|| "stack underflow on HELPER_FUNC".to_string())?;
                let result = apply_helper(instruction.function, &stack[start..])?;
                stack.truncate(start);
                stack.push(result);
            }
            InstructionOp::Gt
            | InstructionOp::Ge
            | InstructionOp::Lt
            | InstructionOp::Le
            | InstructionOp::Eq
            | InstructionOp::Ne => {
                let right = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on comparison".to_string())?;
                let left = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on comparison".to_string())?;
                stack.push(compare(&left, &right, instruction.op)?);
            }
        }
    }
    if stack.len() != 1 {
        return Err("expression evaluation did not end with a single stack value".to_string());
    }
    stack
        .pop()
        .ok_or_else(|| "expression evaluation produced no result".to_string())
}

fn scalar_arg(args: &[Complex64], index: usize) -> f64 {
    let value = args[index];
    debug_assert!(value.im.abs() <= 1e-12);
    value.re
}

fn apply_helper_scalar(function_id: i64, args: &[Complex64]) -> Result<Complex64, String> {
    let value = match function_id {
        1 => Complex64::new(
            gaussian_2d(
                scalar_arg(args, 0),
                scalar_arg(args, 1),
                scalar_arg(args, 2),
                scalar_arg(args, 3),
                scalar_arg(args, 4),
                scalar_arg(args, 5),
                scalar_arg(args, 6),
            ),
            0.0,
        ),
        2 => Complex64::new(
            gaussian_2d_rotated(
                scalar_arg(args, 0),
                scalar_arg(args, 1),
                scalar_arg(args, 2),
                scalar_arg(args, 3),
                scalar_arg(args, 4),
                scalar_arg(args, 5),
                scalar_arg(args, 6),
                scalar_arg(args, 7),
            ),
            0.0,
        ),
        3 => phase_modulation(
            scalar_arg(args, 0),
            scalar_arg(args, 1),
            scalar_arg(args, 2),
        ),
        4 => Complex64::new(
            square_wave(
                scalar_arg(args, 0),
                scalar_arg(args, 1),
                scalar_arg(args, 2),
            ),
            0.0,
        ),
        5 => resonant_polarization_modulation(
            scalar_arg(args, 0),
            scalar_arg(args, 1),
            scalar_arg(args, 2),
        ),
        6 => Complex64::new(
            sawtooth_wave(
                scalar_arg(args, 0),
                scalar_arg(args, 1),
                scalar_arg(args, 2),
            ),
            0.0,
        ),
        7 => Complex64::new(
            variable_on_off(
                scalar_arg(args, 0),
                scalar_arg(args, 1),
                scalar_arg(args, 2),
                scalar_arg(args, 3),
            ),
            0.0,
        ),
        8 | 10 | 14 | 16 => {
            return Err("tuple-valued helper used in scalar evaluator".to_string());
        }
        15 => Complex64::new(
            gaussian_1d(
                scalar_arg(args, 0),
                scalar_arg(args, 1),
                scalar_arg(args, 2),
            ),
            0.0,
        ),
        9 => Complex64::new(
            rabi_from_intensity(
                scalar_arg(args, 0),
                scalar_arg(args, 1),
                if args.len() > 2 {
                    scalar_arg(args, 2)
                } else {
                    ED_XTB
                },
            ),
            0.0,
        ),
        11 => {
            let intensity = gaussian_2d(
                scalar_arg(args, 0),
                scalar_arg(args, 1),
                scalar_arg(args, 2),
                scalar_arg(args, 3),
                scalar_arg(args, 4),
                scalar_arg(args, 5),
                scalar_arg(args, 6),
            );
            Complex64::new(
                rabi_from_intensity(
                    intensity,
                    scalar_arg(args, 7),
                    if args.len() > 8 {
                        scalar_arg(args, 8)
                    } else {
                        ED_XTB
                    },
                ),
                0.0,
            )
        }
        12 => Complex64::new(
            variable_on_off_duty(
                scalar_arg(args, 0),
                scalar_arg(args, 1),
                scalar_arg(args, 2),
                scalar_arg(args, 3),
            ),
            0.0,
        ),
        13 => {
            let n =
                ((scalar_arg(args, 0) - scalar_arg(args, 1)) / scalar_arg(args, 2)).floor() as i64;
            Complex64::new(if n % 2 == 0 { 1.0 } else { -1.0 }, 0.0)
        }
        _ => return Err(format!("unknown helper function id {}", function_id)),
    };
    Ok(value)
}

pub fn eval_scalar_expression_into(
    expression: &CompiledExpression,
    slots: &[RuntimeValue],
    t: f64,
    temps: &[RuntimeValue],
    stack: &mut Vec<Complex64>,
) -> Result<Complex64, String> {
    stack.clear();
    for instruction in &expression.instructions {
        match instruction.op {
            InstructionOp::Const => stack.push(instruction.value),
            InstructionOp::Slot => stack.push(as_scalar(&slots[instruction.index])?),
            InstructionOp::Temp => stack.push(as_scalar(&temps[instruction.index])?),
            InstructionOp::Time => stack.push(Complex64::new(t, 0.0)),
            InstructionOp::Add => {
                let right = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on ADD".to_string())?;
                let left = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on ADD".to_string())?;
                stack.push(left + right);
            }
            InstructionOp::Sub => {
                let right = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on SUB".to_string())?;
                let left = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on SUB".to_string())?;
                stack.push(left - right);
            }
            InstructionOp::Mul => {
                let right = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on MUL".to_string())?;
                let left = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on MUL".to_string())?;
                stack.push(left * right);
            }
            InstructionOp::Div => {
                let right = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on DIV".to_string())?;
                let left = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on DIV".to_string())?;
                stack.push(left / right);
            }
            InstructionOp::Pow => {
                let right = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on POW".to_string())?;
                let left = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on POW".to_string())?;
                stack.push(left.powc(right));
            }
            InstructionOp::Neg => {
                let value = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on NEG".to_string())?;
                stack.push(-value);
            }
            InstructionOp::Conj => {
                let value = stack
                    .pop()
                    .ok_or_else(|| "stack underflow on CONJ".to_string())?;
                stack.push(value.conj());
            }
            InstructionOp::BuiltinFunc => {
                let argc = instruction.argc;
                let start = stack
                    .len()
                    .checked_sub(argc)
                    .ok_or_else(|| "stack underflow on BUILTIN_FUNC".to_string())?;
                let result = apply_builtin_scalar(instruction.function, stack[start])?;
                stack.truncate(start);
                stack.push(result);
            }
            InstructionOp::HelperFunc => {
                let argc = instruction.argc;
                let start = stack
                    .len()
                    .checked_sub(argc)
                    .ok_or_else(|| "stack underflow on HELPER_FUNC".to_string())?;
                let result = apply_helper_scalar(instruction.function, &stack[start..])?;
                stack.truncate(start);
                stack.push(result);
            }
            InstructionOp::Gt
            | InstructionOp::Ge
            | InstructionOp::Lt
            | InstructionOp::Le
            | InstructionOp::Eq
            | InstructionOp::Ne => {
                let right = scalar_arg(stack.as_slice(), stack.len() - 1);
                stack.pop();
                let left = scalar_arg(stack.as_slice(), stack.len() - 1);
                stack.pop();
                let result = match instruction.op {
                    InstructionOp::Gt => left > right,
                    InstructionOp::Ge => left >= right,
                    InstructionOp::Lt => left < right,
                    InstructionOp::Le => left <= right,
                    InstructionOp::Eq => left == right,
                    InstructionOp::Ne => left != right,
                    _ => unreachable!(),
                };
                stack.push(Complex64::new(if result { 1.0 } else { 0.0 }, 0.0));
            }
        }
    }
    if stack.len() != 1 {
        return Err("expression evaluation did not end with a single stack value".to_string());
    }
    stack
        .pop()
        .ok_or_else(|| "expression evaluation produced no result".to_string())
}

pub fn scalar_value(value: RuntimeValue) -> Result<Complex64, String> {
    as_scalar(&value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_op_values_match_python() {
        assert_eq!(InstructionOp::Const as u8, 1);
        assert_eq!(InstructionOp::Slot as u8, 2);
        assert_eq!(InstructionOp::Temp as u8, 3);
        assert_eq!(InstructionOp::Time as u8, 4);
        assert_eq!(InstructionOp::Add as u8, 5);
        assert_eq!(InstructionOp::Sub as u8, 6);
        assert_eq!(InstructionOp::Mul as u8, 7);
        assert_eq!(InstructionOp::Div as u8, 8);
        assert_eq!(InstructionOp::Pow as u8, 9);
        assert_eq!(InstructionOp::Neg as u8, 10);
        assert_eq!(InstructionOp::Conj as u8, 11);
        assert_eq!(InstructionOp::BuiltinFunc as u8, 12);
        assert_eq!(InstructionOp::HelperFunc as u8, 13);
        assert_eq!(InstructionOp::Gt as u8, 14);
        assert_eq!(InstructionOp::Ge as u8, 15);
        assert_eq!(InstructionOp::Lt as u8, 16);
        assert_eq!(InstructionOp::Le as u8, 17);
        assert_eq!(InstructionOp::Eq as u8, 18);
        assert_eq!(InstructionOp::Ne as u8, 19);
    }

    #[test]
    fn instruction_op_roundtrip() {
        for i in 1..=19i64 {
            let op = InstructionOp::from_i64(i).unwrap();
            assert_eq!(op as u8, i as u8);
        }
    }

    fn complex_vec(values: &[f64]) -> Vec<Complex64> {
        values.iter().map(|&v| Complex64::new(v, 0.0)).collect()
    }

    /// Reference values from `scipy.interpolate.PchipInterpolator(grid, values,
    /// extrapolate=True)`, covering both extrapolated ends, the edge intervals
    /// (three-point end slopes), sign changes and the two-point linear case.
    #[test]
    fn pchip_matches_scipy_including_edges_and_extrapolation() {
        let cases: [(&[f64], &[f64], &[f64], &[f64]); 4] = [
            (
                &[0.0, 0.5, 1.7, 2.0, 3.5, 4.0],
                &[0.0, 1.0, 0.2, 0.25, 2.0, -1.0],
                &[-0.7, 0.0, 0.1, 0.5, 1.2, 1.9, 2.6, 3.9, 4.0, 4.6],
                &[
                    0.14164705882352768,
                    0.0,
                    0.2821960784313726,
                    1.0,
                    0.500925925925926,
                    0.22592592592592592,
                    0.92,
                    -0.18933333333333224,
                    -0.9999999999999996,
                    -0.29900000000000304,
                ],
            ),
            (
                &[0.0, 1.0, 3.0],
                &[1.0, 4.0, 2.0],
                &[
                    -0.5249999999999999,
                    0.0,
                    0.07500000000000001,
                    0.375,
                    0.8999999999999999,
                    1.4249999999999998,
                    1.9500000000000002,
                    2.925,
                    3.0,
                    3.4499999999999997,
                ],
                &[
                    -0.9419531249999994,
                    1.0,
                    1.326171875,
                    2.583984375,
                    3.955,
                    3.98080859375,
                    3.7856562499999997,
                    2.2166679687500004,
                    2.0,
                    0.3234687500000013,
                ],
            ),
            (
                &[0.0, 1.0, 3.0],
                &[0.0, 3.0, 3.5],
                &[
                    -0.5249999999999999,
                    0.0,
                    0.07500000000000001,
                    0.375,
                    0.8999999999999999,
                    1.4249999999999998,
                    1.9500000000000002,
                    2.925,
                    3.0,
                    3.4499999999999997,
                ],
                &[
                    -1.6473520784198108,
                    0.0,
                    0.29678294516509435,
                    1.4781747494103774,
                    2.9099858490566035,
                    3.1924083505306604,
                    3.364657724056604,
                    3.49932241303066,
                    3.5,
                    3.4758576061320756,
                ],
            ),
            (
                &[-1.0, 2.0],
                &[3.0, -1.0],
                &[
                    -1.525,
                    -1.0,
                    -0.925,
                    -0.625,
                    -0.10000000000000009,
                    0.4249999999999998,
                    0.9500000000000002,
                    1.9249999999999998,
                    2.0,
                    2.4499999999999997,
                ],
                &[
                    3.6999999999999997,
                    3.0,
                    2.9,
                    2.5,
                    1.8000000000000003,
                    1.1000000000000003,
                    0.3999999999999999,
                    -0.8999999999999995,
                    -1.0,
                    -1.5999999999999996,
                ],
            ),
        ];
        for (grid, values, xs, expected) in cases {
            let grid = complex_vec(grid);
            let values = complex_vec(values);
            for (&x, &want) in xs.iter().zip(expected) {
                let got = pchip_interp(x, &grid, &values).unwrap();
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "x = {x}: got {got}, scipy {want}"
                );
            }
        }
    }

    #[test]
    fn pchip_rejects_unsorted_grid() {
        let grid = complex_vec(&[0.0, 2.0, 1.0]);
        let values = complex_vec(&[0.0, 1.0, 2.0]);
        assert!(pchip_interp(0.5, &grid, &values).is_err());
    }

    fn ins(op: InstructionOp, index: usize, argc: usize, function: i64) -> Instruction {
        Instruction {
            op,
            index,
            argc,
            function,
            value: Complex64::ZERO,
        }
    }

    fn expr(instructions: Vec<Instruction>) -> CompiledExpression {
        CompiledExpression {
            instructions,
            scalar_only: true,
            output_is_tuple: false,
        }
    }

    #[test]
    fn validate_accepts_well_formed_expression() {
        let e = expr(vec![
            ins(InstructionOp::Time, 0, 0, 0),
            ins(InstructionOp::Slot, 1, 0, 0),
            ins(InstructionOp::Const, 0, 0, 0),
            ins(InstructionOp::HelperFunc, 0, 3, 4),
            ins(InstructionOp::BuiltinFunc, 0, 1, 1),
        ]);
        assert!(e.validate(2, 0).is_ok());
    }

    #[test]
    fn validate_rejects_malformed_expressions() {
        // slot out of range
        assert!(expr(vec![ins(InstructionOp::Slot, 3, 0, 0)])
            .validate(2, 0)
            .is_err());
        // temp out of range
        assert!(expr(vec![ins(InstructionOp::Temp, 0, 0, 0)])
            .validate(2, 0)
            .is_err());
        // builtin with no argument
        assert!(expr(vec![ins(InstructionOp::BuiltinFunc, 0, 0, 1)])
            .validate(2, 0)
            .is_err());
        // helper with the wrong arity (multipass_2d_rabi needs 8 or 9)
        let mut short = vec![ins(InstructionOp::Const, 0, 0, 0); 7];
        short.push(ins(InstructionOp::HelperFunc, 0, 7, 10));
        assert!(expr(short).validate(2, 0).is_err());
        // comparison on an empty stack
        assert!(expr(vec![ins(InstructionOp::Gt, 0, 0, 0)])
            .validate(2, 0)
            .is_err());
        // two values left on the stack
        assert!(expr(vec![
            ins(InstructionOp::Time, 0, 0, 0),
            ins(InstructionOp::Time, 0, 0, 0)
        ])
        .validate(2, 0)
        .is_err());
    }

    #[test]
    fn instruction_op_unknown_value() {
        assert!(InstructionOp::from_i64(0).is_err());
        assert!(InstructionOp::from_i64(20).is_err());
        assert!(InstructionOp::from_i64(-1).is_err());
    }
}
