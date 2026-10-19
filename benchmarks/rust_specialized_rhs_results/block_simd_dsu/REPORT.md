# Blockwise and SIMD-preserving D+S+U tests

Eight randomized interleaved rounds per system. All times include full Tsit5 solves over 10 us with the same current prepared H/DAG, full density coordinates, diagonal split, switching segmentation, tolerances and output. Positive reductions mean faster than unmodified D+S+U. The hooked control measures dispatch/layout overhead.

| System | Kernel | Median seconds | Paired runtime reduction | Descriptive 95% interval |
|---|---|---:|---:|---:|
| rc | production | 2.727575 | +0.00% | [+0.00, +0.00]% |
| rc | hooked_control | 2.739702 | -1.27% | [-2.21, +2.76]% |
| rc | blocks | 2.727631 | -0.30% | [-1.20, +3.10]% |
| rc | simd | 2.154029 | +21.18% | [+18.44, +22.53]% |
| rc | both | 2.188911 | +19.49% | [+18.34, +19.93]% |
| r2 | production | 0.262983 | +0.00% | [+0.00, +0.00]% |
| r2 | hooked_control | 0.272119 | -2.96% | [-5.19, -0.61]% |
| r2 | blocks | 0.271860 | -3.02% | [-15.24, -1.30]% |
| r2 | simd | 0.234030 | +11.21% | [+5.95, +13.21]% |
| r2 | both | 0.441579 | -67.56% | [-68.89, -61.69]% |

The SIMD candidate preserves the existing sparse CSR/CSC outer traversal, skips the Hamiltonian diagonal in sparse products, retains the separate diagonal commutator, and accumulates real/imaginary H-rho products into separate scratch arrays. AVX2 handles four consecutive complex density entries at a time. There is no FMA or changed reduction order across Hamiltonian couplings. Runtime feature detection gates the AVX2 helper; unsupported CPUs use its scalar loop.

The block candidate uses the union Hamiltonian/decay coordinate closure to precompute contiguous reachable column runs for each density row. The D+S+U sparse matrix product traverses only those runs and reconstructs only reachable off-diagonal commutator entries. Full density evolution, dissipator and packing remain in place. Unlike the previous reduction prototype, it genuinely skips commutator multiply operations. RC starts with uniform X,J=2 population; R(2) with its fixture uniform X population. Safe closures retain 4360/4900 and 727/1681 real coordinates. No additional magnitude threshold is applied.

All five variants pass complex density-entry accuracy checks against the current unmodified strict reference, including coherences: 1% relative above magnitude 1e-6, absolute 1e-6 below, trace error <1e-9. Three random compatible PSD densities are checked at four EOM times. Every candidate retains the same number of RHS evaluations per system; the timing gains are kernel gains. Candidate atol/rtol=1e-11/1e-9; reference=1e-13/1e-11.

Fixtures are 10 us, E=200 V/cm and B=0.01 G, optical EOM beta=3.8 at 1.6 MHz. RC includes the existing four configurations from 1 MHz microwave switches with a quarter-period offset; boundaries are segmented at 250 ns. These are not full production apparatus transits. No library defaults or installed kernels changed.

The block test does not establish that all block algorithms fail: this particular contiguous-run traversal does not repay its dispatch/slicing overhead on these fixtures. Combining short runs with the AVX2 helper is especially unfavorable for R(2). The standalone SIMD candidate is the useful result to consider integrating, after normal cross-system library validation.

Untested follow-up reminder: fusing diagonal evolution, decay updates and output packing to reduce separate memory traversals.

Reproducer: ../../check_block_simd_dsu.py; kernel: ../../block_simd_dsu.rs. Raw timings, full-density checks, execution orders, source hashes and summaries: results.json. This experiment was implemented and run directly by the main agent, without delegated agents.
