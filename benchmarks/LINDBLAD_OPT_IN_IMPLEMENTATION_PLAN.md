# Opt-in integration of benchmarked Lindblad methods

User authorization: implement the successful techniques now so they can be used in known advantageous cases. **Do not change any defaults.** The broader [test plan](LINDBLAD_KERNEL_PROMOTION_TEST_PLAN.md), including its provisional 1-2 working-day estimate, is saved for later execution; no new broad benchmarks are started here.

Status: implemented and verified locally, 2026-10-11. All eight stable forward aliases are available; native adjoint response and parameter-batch APIs are exported. Existing static dense APIs are retained. Full Lindblad regression suite: **313 passed, 2 skipped**. Ruff passed for new API/tests. Native extension release SHA256: `8AE86750BFC7514FF91B0BB634C0DF10CE07522FD6FECE3E652FD98FF148AE77`. Historical benchmark native hashes and artifacts were not rewritten. The release DLL was copied to the editable package after maturin's pip reinstall step failed on the existing site-packages `.pth` file; imports were verified from outside the repository as well as within it.

Adjoint validation includes analytic spontaneous decay, coherent initial/terminal matrices, independent direct complex-Liouvillian matrix exponentiation, cascade decay plus projector dephasing, exact switch segmentation, smooth complex phase modulation at nonzero absolute entry time, and ordered parallel parameter overrides. It supports constant diagonal running weights and arbitrary Hermitian terminal matrices with the current single-entry collapse representation. It is not a general-collapse or full-density solver. Its native performance has not yet been benchmarked; prior adjoint performance results describe the earlier prototype.

## Implementation scope

1. Add stable explicit execution-mode names for the sparse-Hermitian/dense-density base and all D/S/U combinations. Preserve experimental names as compatible aliases to the identical native implementations. Do not change arithmetic or silently select a different kernel. Validate single, batch and grid acceptance and numerical equivalence. Keep the existing execution-mode, solver and tolerance defaults.
2. Add an explicit package adjoint-response API, backed by Rust, for linear terminal observables and time-integrated emission objectives. Return observable response matrices, basis-state responses and contractions with population mixtures or coherent density inputs. This does not reconstruct final density matrices and must not automatically replace ordinary forward output requests.
3. Preserve absolute physical time and explicit discontinuity schedules when integrating adjoints backward. Validate Hamiltonian sign, reversed dissipative transfer, packed Hermitian metric, running objective and parameter overrides against independent algebra and forward solutions. Reject unsupported plan types/objectives clearly. Keep current benchmark-prototype timings labeled as prototype evidence; do not assert the new native implementation has those same timings without measuring it.
4. Retain the existing static dense APIs and factorization reuse. Explain when changing parameters requires refactorization and how its static restriction differs from adjoint eligibility.
5. Add usage documentation and focused regression tests. Build the extension once after agents finish source edits, then verify new opt-in APIs and existing single/batch/grid behavior. Source-only alias tests are insufficient: run the rebuilt native extension and record its new identity without overwriting benchmark provenance.

## Completion and deferred work

Completion requires usable imports, native dispatch, correct outputs, explicit applicability/error behavior, no default changes, passing focused tests and the relevant existing solver regressions. Do not publish, commit, rebuild benchmark artifacts or run large molecular timing sweeps as part of this implementation.

The larger general-system suite, sampled-output performance comparisons, default promotion decision, serrodyne segmentation and static-plus-time-dependent correction methods remain separate subsequent work.
