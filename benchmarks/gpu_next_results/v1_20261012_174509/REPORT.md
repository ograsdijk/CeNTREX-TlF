# RC and R(2): further GPU techniques

Stage: complete. No suite time cap. No production changes or CPU ODE reruns.

Output-specific timings include the requested GPU-to-host transfer; diagnostics and disk writes are separate. Full density/coherence target is 1% for reference magnitudes >=1e-6; near-zero absolute errors are separate.

| System | Kind | Method | Batch | Step ns | Threads | Tile | Wave table | Seconds | Entry error % | Pass |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| rc | short | tsit5 | 512 | 4.4643 | 64 | 0 | False | 4.429032099986216 | 0.92509 | True |
| rc | short | tsit5 | 512 | 4.4643 | 128 | 0 | False | 4.426651200017659 | 0.92509 | True |
| rc | short | tsit5 | 512 | 4.4643 | 256 | 0 | False | 4.452657200017711 | 0.92509 | True |
| rc | short | tsit5 | 512 | 4.4643 | 512 | 0 | False | 4.503552999987733 | 0.92509 | True |
| rc | short | tsit5 | 512 | 4.4643 | 128 | 16 | False | 5.023894099984318 | 0.92509 | True |
| rc | short | tsit5 | 512 | 4.4643 | 256 | 16 | False | 5.0286865000089165 | 0.92509 | True |
| rc | short | tsit5 | 512 | 4.4643 | 512 | 16 | False | 5.0384190999902785 | 0.92509 | True |
| rc | short | tsit5 | 512 | 4.4643 | 128 | 32 | False | 4.455607699987013 | 0.92509 | True |
| rc | short | tsit5 | 512 | 4.4643 | 256 | 32 | False | 4.19457659998443 | 0.92509 | True |
| rc | short | tsit5 | 512 | 4.4643 | 512 | 32 | False | 4.205825800017919 | 0.92509 | True |
| r2 | short | tsit5 | 512 | 2 | 64 | 0 | False | 0.5726277999929152 | 0.059308 | True |
| r2 | short | tsit5 | 512 | 2 | 128 | 0 | False | 0.5757620000222232 | 0.059308 | True |
| r2 | short | tsit5 | 512 | 2 | 256 | 0 | False | 0.5801136000081897 | 0.059308 | True |
| r2 | short | tsit5 | 512 | 2 | 512 | 0 | False | 0.5900114999967627 | 0.059308 | True |
| r2 | short | tsit5 | 512 | 2 | 128 | 16 | False | 0.6507385000004433 | 0.059308 | True |
| r2 | short | tsit5 | 512 | 2 | 256 | 16 | False | 0.646822799986694 | 0.059308 | True |
| r2 | short | tsit5 | 512 | 2 | 512 | 16 | False | 0.6584032000100706 | 0.059308 | True |
| r2 | short | tsit5 | 512 | 2 | 128 | 32 | False | 0.5478290000173729 | 0.059308 | True |
| r2 | short | tsit5 | 512 | 2 | 256 | 32 | False | 0.5461537000082899 | 0.059308 | True |
| r2 | short | tsit5 | 512 | 2 | 512 | 32 | False | 0.5516942999965977 | 0.059308 | True |
| rc | short | ck54 | 512 | 0.75 | 256 | 32 | False | 20.045736100000795 | 0.098598 | True |
| rc | short | ck54 | 512 | 1 | 256 | 32 | False | 14.997765699983574 | 0.30825 | True |
| rc | short | ck54 | 512 | 1.5 | 256 | 32 | False | 10.038698500022292 | 1.5323 | False |
| rc | short | ck54 | 512 | 2 | 256 | 32 | False | 7.547384100005729 | 4.7907 | False |
| rc | short | ck54 | 512 | 3 | 256 | 32 | False | 5.058718999993289 | 23.467 | False |
| r2 | short | ck54 | 512 | 0.25 | 256 | 32 | False | 3.224283400020795 | 0.00031001 | True |
| r2 | short | ck54 | 512 | 0.5 | 256 | 32 | False | 1.6182472999789752 | 0.0047387 | True |
| r2 | short | ck54 | 512 | 1 | 256 | 32 | False | 0.8111487999849487 | 0.075753 | True |
| r2 | short | ck54 | 512 | 1.5 | 256 | 32 | False | 0.5464194999949541 | 0.3738 | True |
| r2 | short | ck54 | 512 | 2 | 256 | 32 | False | 0.41116089999559335 | 1.0566 | False |
| rc | short | tsit5 | 512 | 4.4643 | 256 | 32 | True | 4.121352399990428 | 0.92509 | True |
| r2 | short | tsit5 | 512 | 2 | 256 | 32 | True | 0.5328317999956198 | 0.059308 | True |
| rc | short | tsit5 | 512 | 0.97656 | 256 | 32 | False | 18.905783000023803 | 0.00036365 | True |
| rc | short | tsit5 | 512 | 4.4643 | 256 | 32 | True | 4.1310265000211075 | 0.92509 | True |
| rc | short | tsit5 | 512 | 4.4643 | 256 | 32 | True | 4.109659699985059 | 0.92509 | True |
| r2 | short | tsit5 | 512 | 0.5 | 256 | 32 | False | 2.1829803999862634 | 0 | True |
| r2 | short | tsit5 | 512 | 2 | 256 | 32 | True | 0.5418457000050694 | 0.059308 | True |
| r2 | short | tsit5 | 512 | 2 | 256 | 32 | True | 0.5337221999943722 | 0.059308 | True |
| rc | short | tsit5 | 1024 | 4.4643 | 256 | 32 | True | 8.716987699997844 | 0.92509 | True |
| rc | short | tsit5 | 2048 | 4.4643 | 256 | 32 | True | 17.695579799998086 | 0.92509 | True |
| rc | short | tsit5 | 4096 | 4.4643 | 256 | 32 | True | 35.4221446000156 | 0.92509 | True |
| r2 | short | tsit5 | 1024 | 2 | 256 | 32 | True | 1.0417918000021018 | 0.059308 | True |
| r2 | short | tsit5 | 2048 | 2 | 256 | 32 | True | 2.287796199991135 | 0.059308 | True |
| r2 | short | tsit5 | 4096 | 2 | 256 | 32 | True | 4.7417022999725305 | 0.059308 | True |
| rc | full | tsit5 | 128 | 4.4643 | 256 | 32 | True | 49.00130970001919 | 9540.6 | False |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 40.6103773000068 | 5869 | False |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 8.234987400006503 | 9929.7 | False |
| r2 | full_reference | tsit5 | 60 | 0.5 | 256 | 32 | False | 69.47351699997671 | 0 | True |
| r2 | full | tsit5 | 60 | 2 | 256 | 32 | True | 16.183705900009954 | 2.3454e-05 | True |
| r2 | full | tsit5 | 60 | 2 | 256 | 32 | True | 15.976992399984738 | 6.8294e-06 | True |
| r2 | full_reference | tsit5 | 60 | 0.5 | 256 | 32 | False | 204.560751299985 | 0 | True |
| r2 | full | tsit5 | 60 | 2 | 256 | 32 | True | 47.45049439999275 | 37.794 | False |
| r2 | full | tsit5 | 60 | 1 | 256 | 32 | True | 94.7096762000001 | 18.927 | False |
| r2 | full_reference | tsit5 | 60 | 0.5 | 256 | 32 | False | 42.03008549998049 | 0 | True |
| r2 | full | tsit5 | 60 | 2 | 256 | 32 | True | 9.854968100000406 | 0.00014696 | True |
| r2 | full | tsit5 | 60 | 2 | 256 | 32 | True | 9.692066299990984 | 8.3168e-06 | True |
| rc | full_reference | tsit5 | 9 | 0.97656 | 256 | 32 | False | 69.50604969999404 | 0.013726 | True |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 14.127583599998616 | 9540.5 | False |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 14.071339699992677 | 0.19937 | True |
| rc | full_reference | tsit5 | 9 | 0.97656 | 256 | 32 | False | 207.37829709998914 | 0.017691 | True |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 42.069103299989365 | 5868.9 | False |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 42.08823750002193 | 0.24455 | True |
| rc | full_reference | tsit5 | 9 | 0.97656 | 256 | 32 | False | 41.89702600002056 | 0.0059978 | True |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 8.51470880000852 | 9929.1 | False |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 8.497616400010884 | 0.065494 | True |
| rc | full | tsit5 | 128 | 4.4643 | 256 | 32 | True | 51.1631924000103 | 0.34343 | True |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 44.73686330000055 | 0.25603 | True |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 8.91644569998607 | 0.31571 | True |
| r2 | short | ck54 | 512 | 1.75 | 256 | 32 | False | 0.47058819999801926 | 0.66637 | True |
| r2 | short | ck54 | 512 | 1.875 | 256 | 32 | False | 0.4398469000007026 | 0.85054 | True |
| r2 | short | ck54 | 512 | 1.9 | 256 | 32 | False | 0.4332834999950137 | 0.88929 | True |
| r2 | short | ck54 | 512 | 1.95 | 256 | 32 | False | 0.4255371000035666 | 0.97304 | True |
| r2 | short | ck54 | 512 | 1.95 | 256 | 32 | True | 0.42314580001402646 | 0.97304 | True |
| r2 | full | ck54 | 60 | 1.95 | 256 | 32 | True | 14.99305980000645 | 0.0048648 | True |
| r2 | full | ck54 | 60 | 1.95 | 256 | 32 | True | 14.827560500008985 | 0.0048782 | True |
| r2 | full | ck54 | 60 | 1.95 | 256 | 32 | True | 44.33615389998886 | 0.0018036 | True |
| r2 | full | ck54 | 60 | 1.95 | 256 | 32 | True | 44.59295619998011 | 0.00076461 | True |
| r2 | full | ck54 | 60 | 1.95 | 256 | 32 | True | 9.176717100024689 | 0.0054911 | True |
| r2 | full | ck54 | 60 | 1.95 | 256 | 32 | True | 8.995980299980147 | 0.0018726 | True |
| rc | short | tsit5 | 512 | 4.4643 | 256 | 32 | True | 4.155334799987031 | 0.92509 | True |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 15.345620500011137 | 0.33849 | True |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 15.29316070000641 | 0.19937 | True |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 45.79576680000173 | 0.24425 | True |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 45.826590399985434 | 0.24455 | True |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 9.221647900005337 | 0.3136 | True |
| rc | full | tsit5 | 9 | 4.4643 | 256 | 32 | True | 9.188874499988742 | 0.065494 | True |
| r2 | short | tsit5 | 512 | 2 | 256 | 32 | True | 0.542155900009675 | 0.059308 | True |
| r2 | full | tsit5 | 60 | 2 | 256 | 32 | True | 17.450398399989353 | 2.3454e-05 | True |
| r2 | full | tsit5 | 60 | 2 | 256 | 32 | True | 17.25584009999875 | 6.8294e-06 | True |
| r2 | full | tsit5 | 60 | 2 | 256 | 32 | True | 52.11715340000228 | 7.5483e-05 | True |
| r2 | full | tsit5 | 60 | 2 | 256 | 32 | True | 51.56137250000029 | 7.7339e-06 | True |
| r2 | full | tsit5 | 60 | 2 | 256 | 32 | True | 10.620260299998336 | 0.00014696 | True |
| r2 | full | tsit5 | 60 | 2 | 256 | 32 | True | 10.456480299995746 | 8.3168e-06 | True |

## Output-specific tests

Photon-only timings exclude density/population transfers. The full density is still evolved internally. Full-reference rows without an external reference are convergence baselines, not independent accuracy checks.

- rc ode_pipeline: complete; speedup 1.0001807089595227. Details: [033_rc_ode_pipeline_tsit5_b512_h4.4643_th256_tile32_wave1.json](033_rc_ode_pipeline_tsit5_b512_h4.4643_th256_tile32_wave1.json).
- r2 ode_pipeline: complete; speedup 1.0026218065160972. Details: [034_r2_ode_pipeline_tsit5_b512_h2_th256_tile32_wave1.json](034_r2_ode_pipeline_tsit5_b512_h2_th256_tile32_wave1.json).
- r2 dense_pipeline: complete; speedup see rows. Details: [035_r2_dense_pipeline_tsit5_b512_h4.4643_th256_tile0_wave0.json](035_r2_dense_pipeline_tsit5_b512_h4.4643_th256_tile0_wave0.json).
- r2 dense_photons: complete; speedup see rows. Details: [036_r2_dense_photons_tsit5_b512_h4.4643_th256_tile0_wave0.json](036_r2_dense_photons_tsit5_b512_h4.4643_th256_tile0_wave0.json).
- rc short joint, v=150 m/s: 4.13103 s, maximum photon relative error 8.7565e-05%.
- rc short photon_only, v=150 m/s: 4.10966 s, maximum photon relative error 8.7565e-05%.
- r2 short joint, v=150 m/s: 0.541846 s, maximum photon relative error 2.4762e-06%.
- r2 short photon_only, v=150 m/s: 0.533722 s, maximum photon relative error 2.4762e-06%.
- r2 full joint, v=150.0 m/s: 16.1837 s, maximum photon relative error 3.30777e-07%.
- r2 full photon_only, v=150.0 m/s: 15.977 s, maximum photon relative error 3.31399e-07%.
- r2 full joint, v=50.0 m/s: 47.4505 s, maximum photon relative error 0.00109789%.
- r2 full joint, v=50.0 m/s: 94.7097 s, maximum photon relative error 0.000739348%.
- r2 full joint, v=250.0 m/s: 9.85497 s, maximum photon relative error 2.7485e-07%.
- r2 full photon_only, v=250.0 m/s: 9.69207 s, maximum photon relative error 2.7485e-07%.
- rc full joint, v=150.0 m/s: 14.1276 s, maximum photon relative error 0.615668%.
- rc full photon_only, v=150.0 m/s: 14.0713 s, maximum photon relative error 4.78429e-05%.
- rc full joint, v=50.0 m/s: 42.0691 s, maximum photon relative error 0.252241%.
- rc full photon_only, v=50.0 m/s: 42.0882 s, maximum photon relative error 5.72928e-05%.
- rc full joint, v=250.0 m/s: 8.51471 s, maximum photon relative error 0.483057%.
- rc full photon_only, v=250.0 m/s: 8.49762 s, maximum photon relative error 2.87452e-05%.
- r2 full joint, v=150.0 m/s: 14.9931 s, maximum photon relative error 2.97897e-05%.
- r2 full photon_only, v=150.0 m/s: 14.8276 s, maximum photon relative error 2.98331e-05%.
- r2 full joint, v=50.0 m/s: 44.3362 s, maximum photon relative error 2.78591e-05%.
- r2 full photon_only, v=50.0 m/s: 44.593 s, maximum photon relative error 2.78591e-05%.
- r2 full joint, v=250.0 m/s: 9.17672 s, maximum photon relative error 3.46704e-05%.
- r2 full photon_only, v=250.0 m/s: 8.99598 s, maximum photon relative error 3.47522e-05%.
- r2 dense_photons: complete; speedup see rows. Details: [084_r2_dense_photons_tsit5_b512_h4.4643_th256_tile0_wave0.json](084_r2_dense_photons_tsit5_b512_h4.4643_th256_tile0_wave0.json).
- rc full joint, v=150.0 m/s: 15.3456 s, maximum photon relative error 4.77836e-05%.
- rc full photon_only, v=150.0 m/s: 15.2932 s, maximum photon relative error 4.78429e-05%.
- rc full joint, v=50.0 m/s: 45.7958 s, maximum photon relative error 5.7259e-05%.
- rc full photon_only, v=50.0 m/s: 45.8266 s, maximum photon relative error 5.72928e-05%.
- rc full joint, v=250.0 m/s: 9.22165 s, maximum photon relative error 2.8714e-05%.
- rc full photon_only, v=250.0 m/s: 9.18887 s, maximum photon relative error 2.87452e-05%.
- r2 full joint, v=150.0 m/s: 17.4504 s, maximum photon relative error 3.30777e-07%.
- r2 full photon_only, v=150.0 m/s: 17.2558 s, maximum photon relative error 3.31399e-07%.
- r2 full joint, v=50.0 m/s: 52.1172 s, maximum photon relative error 4.23031e-07%.
- r2 full photon_only, v=50.0 m/s: 51.5614 s, maximum photon relative error 4.23725e-07%.
- r2 full joint, v=250.0 m/s: 10.6203 s, maximum photon relative error 2.7485e-07%.
- r2 full photon_only, v=250.0 m/s: 10.4565 s, maximum photon relative error 2.7485e-07%.
