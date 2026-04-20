use num_complex::Complex64;

#[derive(Clone, Debug)]
pub struct PackedHermitianLayout {
    pub n: usize,
}

#[derive(Clone, Debug)]
pub struct UpperTriLayout {
    pub n: usize,
    row_offsets: Vec<usize>,
}

impl PackedHermitianLayout {
    pub fn new(n: usize) -> Result<Self, String> {
        if n == 0 {
            return Err("layout dimension must be positive".to_string());
        }
        Ok(Self { n })
    }

    pub fn packed_len(&self) -> usize {
        self.n * self.n
    }

    pub fn pack_into(&self, matrix: &[Complex64], packed: &mut [f64]) -> Result<(), String> {
        if matrix.len() != self.n * self.n {
            return Err(format!(
                "expected {} matrix elements, got {}",
                self.n * self.n,
                matrix.len()
            ));
        }
        if packed.len() != self.packed_len() {
            return Err(format!(
                "expected packed vector length {}, got {}",
                self.packed_len(),
                packed.len()
            ));
        }
        for i in 0..self.n {
            packed[i] = matrix[i * self.n + i].re;
        }
        let mut cursor = self.n;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let value = matrix[i * self.n + j];
                packed[cursor] = value.re;
                packed[cursor + 1] = value.im;
                cursor += 2;
            }
        }
        Ok(())
    }

    pub fn pack(&self, matrix: &[Complex64]) -> Result<Vec<f64>, String> {
        let mut packed = vec![0.0_f64; self.packed_len()];
        self.pack_into(matrix, packed.as_mut_slice())?;
        Ok(packed)
    }

    pub fn unpack_into(&self, packed: &[f64], matrix: &mut [Complex64]) -> Result<(), String> {
        if packed.len() != self.packed_len() {
            return Err(format!(
                "expected packed vector length {}, got {}",
                self.packed_len(),
                packed.len()
            ));
        }
        if matrix.len() != self.n * self.n {
            return Err(format!(
                "expected {} matrix elements, got {}",
                self.n * self.n,
                matrix.len()
            ));
        }
        matrix.fill(Complex64::new(0.0, 0.0));
        for i in 0..self.n {
            matrix[i * self.n + i] = Complex64::new(packed[i], 0.0);
        }
        let mut cursor = self.n;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let value = Complex64::new(packed[cursor], packed[cursor + 1]);
                matrix[i * self.n + j] = value;
                matrix[j * self.n + i] = value.conj();
                cursor += 2;
            }
        }
        Ok(())
    }

    pub fn unpack(&self, packed: &[f64]) -> Result<Vec<Complex64>, String> {
        let mut matrix = vec![Complex64::new(0.0, 0.0); self.n * self.n];
        self.unpack_into(packed, matrix.as_mut_slice())?;
        Ok(matrix)
    }
}

impl UpperTriLayout {
    pub fn new(n: usize) -> Result<Self, String> {
        if n == 0 {
            return Err("upper-triangle layout dimension must be positive".to_string());
        }
        let mut row_offsets = Vec::with_capacity(n);
        let mut offset = 0_usize;
        for row in 0..n {
            row_offsets.push(offset);
            offset += n - row;
        }
        Ok(Self { n, row_offsets })
    }

    pub fn len(&self) -> usize {
        self.n * (self.n + 1) / 2
    }

    pub fn index(&self, i: usize, j: usize) -> Result<usize, String> {
        if i >= self.n || j >= self.n {
            return Err(format!(
                "upper-triangle index ({}, {}) out of bounds for size {}",
                i, j, self.n
            ));
        }
        if i > j {
            return Err(format!(
                "expected upper-triangle index with i <= j, got ({}, {})",
                i, j
            ));
        }
        Ok(self.row_offsets[i] + (j - i))
    }

    pub fn clear(&self, upper: &mut [Complex64]) -> Result<(), String> {
        if upper.len() != self.len() {
            return Err(format!(
                "expected upper-triangle buffer length {}, got {}",
                self.len(),
                upper.len()
            ));
        }
        upper.fill(Complex64::new(0.0, 0.0));
        Ok(())
    }

    #[inline]
    pub fn index_unchecked(&self, i: usize, j: usize) -> usize {
        self.row_offsets[i] + (j - i)
    }

    #[inline]
    pub fn get_hermitian(&self, upper: &[Complex64], i: usize, j: usize) -> Complex64 {
        if i <= j {
            upper[self.index_unchecked(i, j)]
        } else {
            upper[self.index_unchecked(j, i)].conj()
        }
    }

    pub fn pack_from_dense(&self, dense: &[Complex64], upper: &mut [Complex64]) -> Result<(), String> {
        if dense.len() != self.n * self.n {
            return Err(format!(
                "expected dense matrix length {}, got {}",
                self.n * self.n,
                dense.len()
            ));
        }
        if upper.len() != self.len() {
            return Err(format!(
                "expected upper-triangle buffer length {}, got {}",
                self.len(),
                upper.len()
            ));
        }
        for i in 0..self.n {
            for j in i..self.n {
                upper[self.index_unchecked(i, j)] = dense[i * self.n + j];
            }
        }
        Ok(())
    }

    pub fn unpack_packed_state(&self, packed: &[f64], upper: &mut [Complex64]) -> Result<(), String> {
        let expected = self.n * self.n;
        if packed.len() != expected {
            return Err(format!(
                "expected packed vector length {}, got {}",
                expected,
                packed.len()
            ));
        }
        if upper.len() != self.len() {
            return Err(format!(
                "expected upper-triangle buffer length {}, got {}",
                self.len(),
                upper.len()
            ));
        }
        for i in 0..self.n {
            upper[self.index_unchecked(i, i)] = Complex64::new(packed[i], 0.0);
        }
        let mut cursor = self.n;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                upper[self.index_unchecked(i, j)] =
                    Complex64::new(packed[cursor], packed[cursor + 1]);
                cursor += 2;
            }
        }
        Ok(())
    }

    pub fn pack_packed_state(&self, upper: &[Complex64], packed: &mut [f64]) -> Result<(), String> {
        let expected = self.n * self.n;
        if packed.len() != expected {
            return Err(format!(
                "expected packed vector length {}, got {}",
                expected,
                packed.len()
            ));
        }
        if upper.len() != self.len() {
            return Err(format!(
                "expected upper-triangle buffer length {}, got {}",
                self.len(),
                upper.len()
            ));
        }
        for i in 0..self.n {
            packed[i] = upper[self.index_unchecked(i, i)].re;
        }
        let mut cursor = self.n;
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                let value = upper[self.index_unchecked(i, j)];
                packed[cursor] = value.re;
                packed[cursor + 1] = value.im;
                cursor += 2;
            }
        }
        Ok(())
    }

    pub fn expand_to_dense(&self, upper: &[Complex64], dense: &mut [Complex64]) -> Result<(), String> {
        if upper.len() != self.len() {
            return Err(format!(
                "expected upper-triangle buffer length {}, got {}",
                self.len(),
                upper.len()
            ));
        }
        if dense.len() != self.n * self.n {
            return Err(format!(
                "expected dense matrix length {}, got {}",
                self.n * self.n,
                dense.len()
            ));
        }
        dense.fill(Complex64::new(0.0, 0.0));
        for i in 0..self.n {
            for j in i..self.n {
                let value = upper[self.index(i, j)?];
                dense[i * self.n + j] = value;
                if i != j {
                    dense[j * self.n + i] = value.conj();
                }
            }
        }
        Ok(())
    }
}
