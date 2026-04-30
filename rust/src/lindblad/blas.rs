use num_complex::Complex64;
use std::ffi::{c_char, c_void, CString};
use std::iter;
use std::sync::OnceLock;

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_UPPER: i32 = 121;
const CBLAS_NO_TRANS: i32 = 111;

type Hmodule = *mut c_void;
type Zher2kFn = unsafe extern "C" fn(
    layout: i32,
    uplo: i32,
    trans: i32,
    n: i64,
    k: i64,
    alpha: *const c_void,
    a: *const c_void,
    lda: i64,
    b: *const c_void,
    ldb: i64,
    beta: f64,
    c: *mut c_void,
    ldc: i64,
);

unsafe extern "system" {
    fn LoadLibraryW(lp_lib_file_name: *const u16) -> Hmodule;
    fn GetProcAddress(h_module: Hmodule, lp_proc_name: *const c_char) -> *mut c_void;
}

#[derive(Clone, Debug)]
pub struct BlasConfig {
    pub library_path: String,
    pub zher2k_symbol: String,
}

struct LoadedHer2k {
    library_path: String,
    function: Zher2kFn,
}

static ZHER2K: OnceLock<Result<LoadedHer2k, String>> = OnceLock::new();

fn load_zher2k(config: &BlasConfig) -> Result<&'static LoadedHer2k, String> {
    let loaded = ZHER2K.get_or_init(|| unsafe {
        let wide_path: Vec<u16> = config
            .library_path
            .encode_utf16()
            .chain(iter::once(0))
            .collect();
        let module = LoadLibraryW(wide_path.as_ptr());
        if module.is_null() {
            return Err(format!(
                "failed to load BLAS library {}",
                config.library_path
            ));
        }
        let symbol_name = CString::new(config.zher2k_symbol.as_str())
            .map_err(|err| format!("invalid symbol name: {err}"))?;
        let function = GetProcAddress(module, symbol_name.as_ptr());
        if function.is_null() {
            return Err(format!(
                "failed to resolve symbol {} from {}",
                config.zher2k_symbol, config.library_path
            ));
        }
        Ok(LoadedHer2k {
            library_path: config.library_path.clone(),
            function: std::mem::transmute::<*mut c_void, Zher2kFn>(function),
        })
    });
    let loaded = loaded.as_ref().map_err(Clone::clone)?;
    if loaded.library_path != config.library_path {
        return Err(format!(
            "BLAS library already initialized with {}, cannot switch to {}",
            loaded.library_path, config.library_path
        ));
    }
    Ok(loaded)
}

pub fn commutator_her2k(
    config: &BlasConfig,
    hamiltonian: &[Complex64],
    rho: &[Complex64],
    out: &mut [Complex64],
    n: usize,
) -> Result<(), String> {
    if hamiltonian.len() != n * n || rho.len() != n * n || out.len() != n * n {
        return Err("invalid matrix lengths for HER2K commutator".to_string());
    }
    out.fill(Complex64::ZERO);
    let alpha = Complex64::I;
    let loaded = load_zher2k(config)?;
    unsafe {
        (loaded.function)(
            CBLAS_ROW_MAJOR,
            CBLAS_UPPER,
            CBLAS_NO_TRANS,
            n as i64,
            n as i64,
            (&alpha as *const Complex64).cast::<c_void>(),
            rho.as_ptr().cast::<c_void>(),
            n as i64,
            hamiltonian.as_ptr().cast::<c_void>(),
            n as i64,
            0.0,
            out.as_mut_ptr().cast::<c_void>(),
            n as i64,
        );
    }
    for i in 0..n {
        let diag = out[i * n + i];
        out[i * n + i] = Complex64::new(diag.re, 0.0);
        for j in (i + 1)..n {
            let value = out[i * n + j];
            out[j * n + i] = value.conj();
        }
    }
    Ok(())
}
