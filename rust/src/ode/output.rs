use num_complex::Complex64;

pub trait OdeOutput: Send {
    fn push(&mut self, t: f64, y: &[f64]);
    fn times(&self) -> &[f64];
}

#[derive(Clone, Debug)]
pub struct OdeOutputResult {
    pub times: Vec<f64>,
    pub values: OdeOutputValues,
    pub width: usize,
}

#[derive(Clone, Debug)]
pub enum OdeOutputValues {
    Full(Vec<f64>),
    Real(Vec<f64>),
    Complex(Vec<Complex64>),
}

pub struct FullOutput {
    times: Vec<f64>,
    values: Vec<f64>,
    dim: usize,
}

impl FullOutput {
    pub fn new(dim: usize, capacity: usize) -> Self {
        Self {
            times: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity * dim),
            dim,
        }
    }

    pub fn finish(self) -> OdeOutputResult {
        OdeOutputResult {
            times: self.times,
            values: OdeOutputValues::Full(self.values),
            width: self.dim,
        }
    }
}

impl OdeOutput for FullOutput {
    fn push(&mut self, t: f64, y: &[f64]) {
        self.times.push(t);
        self.values.extend_from_slice(y);
    }

    fn times(&self) -> &[f64] {
        &self.times
    }
}

pub struct PopulationsOutput {
    times: Vec<f64>,
    values: Vec<f64>,
    indices: Vec<usize>,
}

impl PopulationsOutput {
    pub fn new(indices: Vec<usize>, capacity: usize) -> Self {
        let width = indices.len();
        Self {
            times: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity * width),
            indices,
        }
    }

    pub fn finish(self) -> OdeOutputResult {
        OdeOutputResult {
            times: self.times,
            values: OdeOutputValues::Real(self.values),
            width: self.indices.len(),
        }
    }
}

impl OdeOutput for PopulationsOutput {
    fn push(&mut self, t: f64, y: &[f64]) {
        self.times.push(t);
        for &idx in &self.indices {
            self.values.push(y[idx]);
        }
    }

    fn times(&self) -> &[f64] {
        &self.times
    }
}

pub struct SelectedOutput {
    times: Vec<f64>,
    values: Vec<Complex64>,
    extractions: Vec<SelectedExtraction>,
}

#[derive(Clone, Debug)]
pub enum SelectedExtraction {
    Real(usize),
    ComplexPair { real_idx: usize, imag_idx: usize },
    ComplexPairConj { real_idx: usize, imag_idx: usize },
}

impl SelectedOutput {
    pub fn new(extractions: Vec<SelectedExtraction>, capacity: usize) -> Self {
        let width = extractions.len();
        Self {
            times: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity * width),
            extractions,
        }
    }

    pub fn finish(self) -> OdeOutputResult {
        OdeOutputResult {
            times: self.times,
            values: OdeOutputValues::Complex(self.values),
            width: self.extractions.len(),
        }
    }
}

impl OdeOutput for SelectedOutput {
    fn push(&mut self, t: f64, y: &[f64]) {
        self.times.push(t);
        for extraction in &self.extractions {
            match extraction {
                SelectedExtraction::Real(idx) => {
                    self.values.push(Complex64::new(y[*idx], 0.0));
                }
                SelectedExtraction::ComplexPair { real_idx, imag_idx } => {
                    self.values.push(Complex64::new(y[*real_idx], y[*imag_idx]));
                }
                SelectedExtraction::ComplexPairConj { real_idx, imag_idx } => {
                    self.values
                        .push(Complex64::new(y[*real_idx], -y[*imag_idx]));
                }
            }
        }
    }

    fn times(&self) -> &[f64] {
        &self.times
    }
}

pub struct FinalOnlyOutput {
    time: Option<f64>,
    state: Vec<f64>,
    dim: usize,
}

impl FinalOnlyOutput {
    pub fn new(dim: usize) -> Self {
        Self {
            time: None,
            state: vec![0.0; dim],
            dim,
        }
    }

    pub fn finish(self) -> OdeOutputResult {
        let mut times = Vec::new();
        let mut values = Vec::new();
        if let Some(t) = self.time {
            times.push(t);
            values.extend_from_slice(&self.state);
        }
        OdeOutputResult {
            times,
            values: OdeOutputValues::Full(values),
            width: self.dim,
        }
    }
}

impl OdeOutput for FinalOnlyOutput {
    fn push(&mut self, t: f64, y: &[f64]) {
        self.time = Some(t);
        self.state.copy_from_slice(y);
    }

    fn times(&self) -> &[f64] {
        match &self.time {
            Some(_) => std::slice::from_ref(self.time.as_ref().unwrap()),
            None => &[],
        }
    }
}

pub struct WeightedIntegralOutput {
    weights: Vec<(usize, f64)>,
    integral: f64,
    last_t: f64,
    last_value: f64,
    times: Vec<f64>,
}

impl WeightedIntegralOutput {
    pub fn new(weights: Vec<(usize, f64)>) -> Self {
        Self {
            weights,
            integral: 0.0,
            last_t: f64::NAN,
            last_value: 0.0,
            times: Vec::new(),
        }
    }

    pub fn finish(self) -> OdeOutputResult {
        OdeOutputResult {
            times: self.times,
            values: OdeOutputValues::Real(vec![self.integral]),
            width: 1,
        }
    }
}

impl OdeOutput for WeightedIntegralOutput {
    fn push(&mut self, t: f64, y: &[f64]) {
        let value: f64 = self.weights.iter().map(|&(i, w)| w * y[i]).sum();
        if self.last_t.is_finite() {
            self.integral += 0.5 * (self.last_value + value) * (t - self.last_t);
        }
        self.last_t = t;
        self.last_value = value;
        if self.times.is_empty() {
            self.times.push(t);
        } else {
            self.times[0] = t;
        }
    }

    fn times(&self) -> &[f64] {
        &self.times
    }
}
