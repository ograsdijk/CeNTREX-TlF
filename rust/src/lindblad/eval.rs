use num_complex::Complex64;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub enum RuntimeValue {
    Scalar(Complex64),
    Tuple(Vec<Complex64>),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
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

fn sawtooth_wave(t: f64, omega: f64, phase: f64) -> f64 {
    ((omega * t + phase - PI) / (2.0 * PI)).rem_euclid(1.0)
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

fn helper_args_to_tuple(arg: &RuntimeValue) -> Result<&[Complex64], String> {
    match arg {
        RuntimeValue::Tuple(values) => Ok(values.as_slice()),
        RuntimeValue::Scalar(_) => Err("helper expected tuple argument".to_string()),
    }
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
                        2.6675506e-30
                    },
                ),
                0.0,
            )))
        }
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
        8 | 10 => {
            return Err("tuple-valued helper used in scalar evaluator".to_string());
        }
        9 => Complex64::new(
            rabi_from_intensity(
                scalar_arg(args, 0),
                scalar_arg(args, 1),
                if args.len() > 2 {
                    scalar_arg(args, 2)
                } else {
                    2.6675506e-30
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
                        2.6675506e-30
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
