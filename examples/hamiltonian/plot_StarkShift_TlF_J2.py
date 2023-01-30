import matplotlib.pyplot as plt
import numpy as np

from centrex_tlf import hamiltonian, states

# generate the hyperfine sublevels in J=0 to J=6
QN = states.generate_uncoupled_states_ground(Js=np.arange(7))

# generate the X hamiltonian terms
H = hamiltonian.generate_uncoupled_hamiltonian_X(QN)

# create a function outputting the hamiltonian as a function of E and B
Hfunc = hamiltonian.generate_uncoupled_hamiltonian_X_function(H)

# V/cm
Ez = np.linspace(0, 50e3, 101)

# generate the Hamiltonian for (almost) zero field, add a small field to make states
# non-degenerate
Hi = Hfunc(E=[0.0, 0.0, 1e-3], B=[0.0, 0.0, 1e-3])
E, V = np.linalg.eigh(Hi)

# get the true superposition-states of the system
QN_states = hamiltonian.matrix_to_states(V, QN)

# original eigenvectors used in tracking states as energies change order
V_track = V.copy()

# indices of the J=2, mJ=0 states focused by the lens
indices_J2_mJ0 = [
    idx for idx, s in enumerate(QN_states) if s.largest.J == 2 and s.largest.mJ == 0
]

indices_J012 = [idx for idx, s in enumerate(QN_states) if s.largest.J in [0, 1, 2]]

# empty array for storing energies
energy = np.empty([Ez.size, len(QN)], dtype=np.complex128)

# iterate over the electric field values
for idx, Ei in enumerate(Ez):
    Hi = Hfunc(E=[0.0, 0.0, Ei], B=[0.0, 0.0, 1e-3])
    E, V = np.linalg.eigh(Hi)

    # sort indices to keep the state order the same
    indices = np.argmax(np.abs(V_track.conj().T @ V), axis=1)
    energy[idx, :] = E[indices]
    V_track[:, :] = V[:, indices]

# plot the J=2, mJ=0 Stark curves
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(
    Ez,
    (energy.real[:, indices_J2_mJ0] - energy.real[:, indices_J2_mJ0][0, 0])
    / (2 * np.pi * 1e9),
    lw=3,
    color="k",
)
ax.set_xlabel("E [V/cm]")
ax.set_ylabel("Energy [GHz]")
ax.set_title("|J=2, mJ=0> Stark Curve")
ax.grid(True)

# plot the J=0,1 and 2 Stark Curves
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(Ez, (energy.real[:, indices_J012]) / (2 * np.pi * 1e9))
ax.set_xlabel("E [V/cm]")
ax.set_ylabel("Energy [GHz]")
ax.set_title("|J=0>, |J=1> and |J=2> Stark Curve")
ax.grid(True)
plt.show()
