"""TEMPORARY CI diagnostic - to be removed. Dumps the intermediate quantities behind the
platform-dependent field-mixed matrix element so they can be compared across runners."""

import numpy as np

from centrex_tlf import hamiltonian, states, transitions


def test_temp_diagnostic():
    import numpy.linalg as nla

    tr = transitions.P2_F1_3o2_F1
    POL = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
    H = hamiltonian.generate_reduced_hamiltonian_transitions(
        [tr],
        E=np.array([0.0, 0.0, 200.0]),
        B=np.array([0.0, 0.0, 1e-5]),
        retain_opposite_parity_levels=True,
        Jmax_X=4,
        Jmax_B=4,
    )
    lines = [f"numpy={np.__version__}", f"nQN={len(H.QN)}"]
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        lines.append(f"blas={blas.get('name')} {blas.get('version')}")
    except Exception as e:
        lines.append(f"blas=? {e}")

    gi = np.asarray(
        states.QuantumSelector(
            electronic=states.ElectronicState.X, J=2, F1=2.5, F=3, mF=-1, P=1
        ).get_indices(H.QN),
        dtype=int,
    ).ravel()[0]
    ei = np.asarray(
        states.QuantumSelector(
            electronic=states.ElectronicState.B, J=1, F1=1.5, F=1, mF=0, P=-1
        ).get_indices(H.QN),
        dtype=int,
    ).ravel()[0]
    gnd, exc = H.QN[gi], H.QN[ei]
    me = abs(
        hamiltonian.generate_ED_ME_mixed_state(exc, gnd, pol_vec=POL, normalize_pol=True)
    )
    lines.append(f"ME={me:.10f}")
    lines.append(f"gi={gi} ei={ei}")
    lines.append(
        "gnd_top=" + ";".join(
            f"{abs(a)**2:.8f}|F={b.F},F1={b.F1},mF={b.mF},J={b.J}"
            for a, b in sorted(gnd.data, key=lambda d: -abs(d[0]) ** 2)[:4]
        )
    )
    lines.append(
        "exc_top=" + ";".join(
            f"{abs(a)**2:.8f}|F={b.F},F1={b.F1},mF={b.mF},P={b.P}"
            for a, b in sorted(exc.data, key=lambda d: -abs(d[0]) ** 2)[:4]
        )
    )
    lines.append(f"Hii_g={H.H_int[gi, gi].real / (2 * np.pi * 1e6):.9f}")
    lines.append(f"Hii_e={H.H_int[ei, ei].real / (2 * np.pi * 1e6):.9f}")
    lines.append(f"trace={np.trace(H.H_int).real:.9e}")
    lines.append(f"fro={nla.norm(H.H_int):.12e}")
    eig = np.sort(nla.eigvalsh(H.H_int).real)
    lines.append(f"eig0={eig[0]:.12e} eigN={eig[-1]:.12e}")
    raise AssertionError("DIAGNOSTIC || " + " || ".join(lines))
