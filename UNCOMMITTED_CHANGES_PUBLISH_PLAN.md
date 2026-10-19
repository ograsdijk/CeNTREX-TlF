# Commit and publish the pending solver work

Status: implemented on `feature/lindblad-solver-improvements`, based on `4eea25f`.
The five implementation commits are `58dd9f5` (CPU/batch outputs), `11cdda7`
(dense propagation), `0c2a0b0` (adjoint), `355cc2a` (CUDA and exact reduction),
and `836fe22` (periodic maps). CUDA integration and reduction share one commit
because their lowering/output paths are coupled. Documentation and curated
benchmark evidence follow in a separate commit.

The isolated committed-source wheel passed 671 CPU tests (25 skipped), all 447
solver tests in the GPU environment, and 66 native Rust tests (one ignored).
See [the validation record](benchmarks/SOLVER_PUBLICATION_VALIDATION.md).
The unrelated physics/model changes and bulk local artifacts remain uncommitted.
Publishing this branch does not merge it or release a package.

## Starting point

- Current branch: `benchmark/rc-propagation`, at `4eea25f`, with no upstream configured.
- The working tree contains earlier CPU/Rust, dense, adjoint, CUDA and physics changes,
  in addition to the latest exact-density reduction and periodic-map APIs.
- Some new modules are still untracked. Committing only the latest files would omit
  dependencies already referenced by the package and its documentation.
- Local GPU virtual environments and CuPy/test caches are currently untracked too;
  these must not be swept into a commit with a blanket `git add .`.

## 1. Inventory and preserve the complete working tree

Create an explicit manifest of modified/untracked files and classify them as:

- Package source, Rust source, dependency declarations and native build requirements.
- Automated tests and public API documentation.
- Benchmark drivers, small result summaries, provenance and supporting fixture inputs.
- Separate physics/model changes: constants, Hamiltonian reduction, Doppler/Rabi utilities,
  field-identification reports and their tests.
- Local environments, compiled kernels, test temporaries, logs and generated bulk data.

Read the actual diffs before assigning ownership or scope. Preserve unrelated work;
do not reset, clean, or delete it. Check existing local feature branches for already
committed versions before duplicating changes. Determine which fixture inputs a
reproducer needs; a script that only references an inaccessible sibling `.cache`
is not a self-contained benchmark. Document or provide a regeneration route.

## 2. Prepare the publication branch and explicit file list

Create `feature/lindblad-solver-improvements` from the current HEAD (or use a new
descriptive name if that branch already exists). Carry the working changes across
without resetting them. Check the current remote base before deciding whether
an update is needed; preserve the local work before any merge/rebase.

Add precise ignore rules for the local GPU environment, CuPy cache and test
temporary directories. Keep source, tests, small JSON/Markdown results and useful
figures available for review. Inspect binary sizes and references before deciding
whether large generated artifacts belong in Git; do not hide all benchmark results.

Keep unrelated physics/model changes out of the solver branch until their scope
and tests have been reviewed. If they are intended for publication, prepare a
separate branch/PR for them while preserving the original working copies.

## 3. Build dependency-ordered, reviewable commits

Proposed boundaries, adjusted where the actual import/native dependencies require it:

1. Rust/prepared-RHS and CPU kernel improvements, public kernel selection and tests.
2. Dense propagation/reconstruction and vectorized packing improvements, with tests.
3. Adjoint capability and its explicit public APIs/tests; no automatic full-density substitution.
4. Optional CUDA fixed-step integration, output handling and dependency extras, with tests.
5. Exact CUDA density-coordinate reduction, opt-in flag and reconstruction/output tests.
6. Periodic CUDA propagators, periodicity/support validation, phase/remainder handling and tests.
7. Curated benchmark evidence and remaining cross-cutting documentation.

Source, exports, configuration, focused tests and essential usage instructions
should land together where possible. Stage explicit paths or hunks, especially
in shared files such as `__init__.py`, `batch.py`, `solve.py` and the README.
Each commit must import/build without relying on untracked files or later commits.
Keep documented defaults accurate: the two latest features remain opt-in.

## 4. Validate the publishable tree

- Rebuild the Rust extension from the staged source rather than relying on the
  currently installed editable binary; verify the declared Rust 1.89 minimum
  and CPU-feature dispatch against the build/CI configuration.
- Run relevant CPU/native tests, lint and whitespace checks. Confirm base-package
  imports and ordinary CPU solves work without CuPy/PyTorch installed.
- Run CUDA/dense/periodic tests in the GPU environment. Existing coverage includes
  full densities/coherences, selected entries, populations, photon integrals/rates,
  snapshots, new initial support, entry phases and fractional cycles.
- Retain the completed RC/R(2) validation evidence. Repeat simulations only if
  publication cleanup changes execution or invalidates that evidence.
- Check packaging includes the new Python/Rust modules and optional CUDA extras.
  Inspect the staged diff/file list for missing imports, private/local paths,
  credentials, environment contents and unintended large artifacts.
- Verify from an isolated checkout of the committed tree that required source and
  tests do not depend on files accidentally left untracked in this workspace.

Record exact commands/results in the branch summary. Do not launch the older
multi-day promotion suite merely to publish already validated opt-in features.

## 5. Push and make the result reviewable

Push the completed feature branch with its upstream set. Verify the remote branch
points at the same commit as local HEAD. Report the branch link, commit IDs,
validation results and any deliberately excluded work. If a PR is requested,
use the final implementation scope and explicitly describe opt-in behavior and
periodic-generator restrictions. Do not merge, tag a release or publish to PyPI
as part of this branch-publication task.
