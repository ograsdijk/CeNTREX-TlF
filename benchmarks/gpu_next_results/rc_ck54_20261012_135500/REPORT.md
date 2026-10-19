# RC and R(2): further GPU techniques

Stage: complete. No suite time cap. No production changes or CPU ODE reruns.

Output-specific timings include the requested GPU-to-host transfer; diagnostics and disk writes are separate. Full density/coherence target is 1% for reference magnitudes >=1e-6; near-zero absolute errors are separate.

| System | Kind | Method | Batch | Step ns | Threads | Tile | Wave table | Seconds | Entry error % | Pass |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| rc | short | ck54 | 512 | 1.2 | 256 | 32 | True | 12.662804399995366 | 0.63418 | True |
| rc | short | ck54 | 512 | 1.3 | 256 | 32 | True | 11.755566499981796 | 0.87096 | True |
| rc | short | ck54 | 512 | 1.35 | 256 | 32 | True | 11.28550389999873 | 1.0119 | False |
| rc | short | ck54 | 512 | 1.4 | 256 | 32 | True | 10.866561600007117 | 1.1665 | False |
| rc | short | ck54 | 512 | 1.3 | 256 | 32 | False | 11.59902960000909 | 0.87096 | True |
| rc | short | ck54 | 512 | 1.3 | 256 | 32 | True | 11.68560460000299 | 0.87096 | True |
| rc | short | ck54 | 512 | 1.3 | 128 | 0 | True | 11.924492900026962 | 0.87096 | True |
| rc | short | ck54 | 512 | 1.3 | 256 | 0 | True | 11.995909599994775 | 0.87096 | True |
| rc | short | tsit5 | 512 | 4.4643 | 256 | 32 | True | 4.152264299977105 | 0.92509 | True |
| rc | full | ck54 | 9 | 1.3 | 256 | 32 | False | 43.649405800009845 | 0.29594 | True |
| rc | full | ck54 | 9 | 1.3 | 256 | 32 | False | 43.67824979999568 | 0.29602 | True |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 15.372315400018124 | 0.33849 | True |
| rc | full | ck54 | 9 | 1.3 | 256 | 32 | False | 130.18185920000542 | 0.35088 | True |
| rc | full | ck54 | 9 | 1.3 | 256 | 32 | False | 130.1692083000089 | 0.35093 | True |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 45.766916699998546 | 0.24425 | True |
| rc | full | ck54 | 9 | 1.3 | 256 | 32 | False | 26.286884400004055 | 0.13714 | True |
| rc | full | ck54 | 9 | 1.3 | 256 | 32 | False | 26.30820749999839 | 0.13735 | True |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 9.18810110000777 | 0.3136 | True |
| rc | full | ck54 | 128 | 1.3 | 256 | 32 | False | 140.10854439999093 | 0.28223 | True |
| rc | full | tsit5 | 128 | 4.4643 | 256 | 32 | True | 50.735863600013545 | 0.34343 | True |

## Output-specific tests

Photon-only timings exclude density/population transfers. The full density is still evolved internally. Full-reference rows without an external reference are convergence baselines, not independent accuracy checks.

- rc full joint, v=150.0 m/s: 43.6494 s, maximum photon relative error 1.76284e-05%.
- rc full photon_only, v=150.0 m/s: 43.6782 s, maximum photon relative error 1.76296e-05%.
- rc full joint, v=150.0 m/s: 15.3723 s, maximum photon relative error 4.77836e-05%.
- rc full joint, v=50.0 m/s: 130.182 s, maximum photon relative error 1.58076e-05%.
- rc full photon_only, v=50.0 m/s: 130.169 s, maximum photon relative error 1.5812e-05%.
- rc full joint, v=50.0 m/s: 45.7669 s, maximum photon relative error 5.7259e-05%.
- rc full joint, v=250.0 m/s: 26.2869 s, maximum photon relative error 1.85909e-05%.
- rc full photon_only, v=250.0 m/s: 26.3082 s, maximum photon relative error 1.86021e-05%.
- rc full joint, v=250.0 m/s: 9.1881 s, maximum photon relative error 2.8714e-05%.
