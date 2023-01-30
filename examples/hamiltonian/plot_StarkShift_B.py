import matplotlib.pyplot as plt
import numpy as np

from centrex_tlf import hamiltonian, states

# generate the hyperfine sublevels in J=0 to J=4
QN = states.generate_coupled_states_excited(
    Js=np.arange(1, 5), Ps=None, Omegas=[-1, 1], basis=states.Basis.CoupledΩ
)


# generate the X hamiltonian terms
H = hamiltonian.generate_coupled_hamiltonian_B(QN)

# create a function outputting the hamiltonian as a function of E and B
Hfunc = hamiltonian.generate_coupled_hamiltonian_B_function(H)

# V/cm
Ez = np.linspace(-1_000, 1_000, 101)

# generate the Hamiltonian for (almost) zero field, add a small field to make states
# non-degenerate
Hi = Hfunc(E=[0, 0, 1e-3], B=[0, 0, 1e-3])
E, V = np.linalg.eigh(Hi)

# get the true superposition-states of the system
QN_states = hamiltonian.matrix_to_states(V, QN)

# original eigenvectors used in tracking states as energies change order
V_track = V.copy()

# empty array for storing energies
energy = np.empty([Ez.size, len(QN)], dtype=np.float64)

# iterate over the electric field values
for idx, Ei in enumerate(Ez):
    Hi = Hfunc(E=[0.0, 0.0, Ei], B=[0.0, 0.0, 0.0])
    E, V = np.linalg.eigh(Hi)

    # sort indices to keep the state order the same
    indices = np.argmax(np.abs(V_track.conj().T @ V), axis=1)
    energy[idx, :] = E[indices]
    V_track[:, :] = V[:, indices]

# indices of the J'=1 states
indices_J1 = [idx for idx, s in enumerate(QN_states) if s.largest.J == 1]


# plot the J'=1 Zeeman Curves
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(
    Ez,
    (energy.real[:, indices_J1] - energy.real[:, indices_J1][0, 0]) / (2 * np.pi * 1e6),
    lw=3,
    color="k",
)
ax.set_xlabel("E [V/cm]")
ax.set_ylabel("Energy [MHz]")
ax.set_title("|J'=1> Stark Curves")
ax.grid(True)
plt.show()
