"""P(2) F1'=3/2 F'=1 at 170 V/cm with Lemont's vertical Earth field.

WMM-2025, 2026-10-05, latitude 41.6736, longitude -88.0017,
elevation 0 km: downward component 48864.4 nT (NOAA NCEI).
Source: https://www.ngdc.noaa.gov/geomag/calculators/igrfwmmForm.shtml
Only the vertical component is included; +z points upward.
The positive electrode is above the negative electrode: Ez = -170 V/cm.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from centrex_tlf import transitions
from centrex_tlf.utils.plotting import plot_transition_level_diagram


def main() -> None:
    diagram = plot_transition_level_diagram(
        transitions.P2_F1_3o2_F1,
        E=-170.0,
        B=-48864.4 / 1e5,
        title="P(2) F'=1 | E = 170 V/cm downward | Lemont vertical Earth field",
    )
    output = Path(__file__).with_name("p2_f1_3o2_f1_lemont_level_diagram_energy_axes.png")
    diagram.fig.savefig(output, dpi=180, bbox_inches="tight")
    diagram.fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(diagram.fig)
    print(f"Saved {output}")
    print(f"Saved {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
