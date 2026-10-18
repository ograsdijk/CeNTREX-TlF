# Solver publication validation

Validated implementation commit: `836fe22`, with dependency commits recorded in
[the publication plan](../UNCOMMITTED_CHANGES_PUBLISH_PLAN.md). Documentation-only
changes follow this implementation. Validation ran on Windows, Python 3.11,
Ryzen 7 9800X3D and RTX 5070 Ti, using Rust 1.98.0 and maturin 1.10.2.

## Isolated package build

`git archive HEAD` supplied a clean source tree under `.cache/publication/verify`.
It contains committed files only, excluding the editable installation, unrelated
working-tree physics changes, and untracked benchmark inputs. The release wheel
was built and installed into an isolated target:

```powershell
$env:CARGO_TARGET_DIR = "$PWD/rust/target"
.venv/Scripts/python.exe -m maturin build --release --offline `
  --interpreter .venv/Scripts/python.exe `
  --manifest-path .cache/publication/verify/rust/Cargo.toml `
  --out .cache/publication/wheels
.venv/Scripts/python.exe -m pip install --no-index --no-deps `
  --target .cache/publication/installed `
  .cache/publication/wheels/centrex_tlf-0.2.10-cp311-cp311-win_amd64.whl
```

Both test environments imported that installed package, not the editable source.
The native extension SHA256 was
`42410756074cb150046256f90702d2e529494cce24282bc762c603b5426bb659`.
Base-package import loaded neither CuPy nor PyTorch. The base environment has
neither optional dependency installed; the GPU environment supplies both.

## Results

| Check | Result |
|---|---|
| Entire Python suite, base environment | 671 passed, 25 skipped |
| `tests/lindblad`, GPU environment | 447 passed |
| Native `cargo test --release --offline --lib` | 66 passed, one ignored |
| `cargo fmt --check` | Passed |
| Ruff on new solver modules and modified preparation/layout modules | Passed |

Python tests used `-q -p no:cacheprovider --tb=short` from the isolated source
directory, with the isolated install first on `sys.path`. As in the repository
CI configuration, `tests/lindblad/test_precomputed_field_basis.py` was excluded.
The runner used a Windows multiprocessing main guard. `CUPY_CACHE_DIR` pointed
to the repository-local ignored cache. Native tests used the isolated source
manifest and the project interpreter through `PYO3_PYTHON`.

The four earlier implementation commits also received separate archived-source
`cargo check --release --offline` and base-package import checks. These import
checks used the final native extension; the complete final source received the
fresh wheel build and both test suites above.

The declared Rust minimum is 1.89. Local checks used 1.98, not 1.89; existing CI
uses stable Rust on Linux/Windows across Python 3.10–3.14. This record does not
claim an additional local minimum-version or cross-platform run.

## Scope and benchmark interpretation

CPU automatic ISA selection affects the packed D+S+U execution path. The default
execution mode remains `expanded_sparse`. CUDA density reduction and periodic
maps remain opt-in; CF4/Magnus remains a benchmark prototype.

[The evidence index](PUBLICATION_INDEX.md) gives measured timings, baselines,
accuracy limits and regeneration instructions. No new physical simulations were
needed for this publication. The historical reports retain their original
provenance, including local artifact paths; the index distinguishes archived
inputs from self-contained package tests.

Constants, Hamiltonian-reduction and Doppler/Rabi utility edits were deliberately
excluded and preserved in the working tree. Virtual environments, compiled
artifacts, caches, logs and large density datasets are not published.
