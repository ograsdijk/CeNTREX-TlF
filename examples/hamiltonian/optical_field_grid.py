"""Frequencies and dressed couplings of X -> B components versus electric field.

Builds a transition field grid for ground J <= 2 over 0-300 V/cm, prints the
components of R(2) F1'=7/2 F'=3 at one field, and plots the P(2) F1'=3/2 F'=1
components versus field, coloured by their dressed strength for X-polarized
light. Parity-forbidden components (upper parent parity equal to the ground
parity) are drawn dashed: they are driven only through Stark mixing.

For ground J <= 8 (X to J=10, B to J=11) the same call takes about 30 s, most
of it building the B-state matrices, and the saved file is about 10 MB.
"""

import matplotlib.pyplot as plt
import numpy as np

from centrex_tlf import transitions
from centrex_tlf.couplings import polarization_X

grid = transitions.build_optical_field_grid(J_ground_max=2, E_max_V_cm=300.0)
print(f"{grid.n_components} components on {grid.E_fields.size} stored fields")
for flag in grid.flags:
    print(f"  flagged {flag.E_low:6.1f}-{flag.E_high:6.1f} V/cm: {flag.reason} ({flag.detail})")

E = 171.6
line = transitions.R2_F1_7o2_F3
print(f"\n{line.name} at {E} V/cm")
for c in sorted(grid.components(line, E), key=lambda c: c.frequency_Hz):
    if c.strength < 1e-3:
        continue
    print(
        f"  {c.lower}  ->  {c.upper}  q={c.q:+d}  "
        f"{c.frequency_Hz / 1e12:.9f} THz  |d|^2={c.strength:.4f}  "
        f"purity {c.lower_purity:.3f}/{c.upper_purity:.3f}"
    )

# P(2) F1'=3/2 F'=1 versus field
line = transitions.P2_F1_3o2_F1
fields = np.linspace(0.0, 300.0, 151)
rows = [grid.at(E_) for E_ in fields]
mask = (rows[0]["label"] == line.name).to_numpy()
freqs = np.array([r["frequency_Hz"].to_numpy()[mask] for r in rows])
weights = np.array([grid.strength(polarization_X, E_)[mask] for E_ in fields])
forbidden = ~rows[0]["parity_allowed"].to_numpy()[mask]

f0 = freqs[0][~forbidden].mean()
fig, ax = plt.subplots(figsize=(8, 6))
wmax = weights.max()
for k in range(freqs.shape[1]):
    if weights[:, k].max() < 1e-3 * wmax:
        continue
    ax.plot(
        fields,
        (freqs[:, k] - f0) / 1e6,
        ls="--" if forbidden[k] else "-",
        lw=0.5 + 2.5 * weights[:, k].max() / wmax,
        color="C1" if forbidden[k] else "C0",
    )
ax.set_xlabel("E [V/cm]")
ax.set_ylabel(f"{line.name} component frequency - zero-field mean [MHz]")
ax.set_title("solid: parity allowed, dashed: Stark-mixing only; width ~ X-pol strength")
ax.grid(True, alpha=0.3)
plt.show()
