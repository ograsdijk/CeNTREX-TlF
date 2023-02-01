[![Python versions on PyPI](https://img.shields.io/pypi/pyversions/centrex-tlf.svg)](https://pypi.python.org/pypi/centrex-tlf/)
[![CeNTREX-TlF version on PyPI](https://img.shields.io/pypi/v/centrex-tlf.svg "CeNTREX-TlF on PyPI")](https://pypi.python.org/pypi/centrex-tlf/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Extensions
![CeNTREX-TlF-julia-extension version on PyPI](https://img.shields.io/badge/extension-julia-informational "CeNTREX-TlF-julia-extension on PyPI") [![CeNTREX-TlF version on PyPI](https://img.shields.io/pypi/v/centrex-tlf-julia-extension.svg "CeNTREX-TlF on PyPI")](https://pypi.python.org/pypi/centrex-tlf-julia-extension/)
# CeNTREX-TlF
Code for generating the CeNTREX TlF States, Hamiltonians, Transitions, Couplings and Lindblad equations.

Consists of six modules:
* `states`
* `hamiltonian`
* `transitions`
* `couplings`
* `lindblad`
* `utils`

`states` has code to generate states and the classes that describe the `CoupledBasisState`, `UncoupledBasisState` and `State`; where `State` holds multiple `CoupledBasisStates` or `UncoupledBasisStates` with different amplitudes, i.e. when superpositions arise.

## Dependencies
* `numpy`
* `scipy`
* `sympy`
* `pandas`
  
## Installation
`python -m pip install .`  
where `.` is the path to the directory. To install directly from `Github` use:  
`python -m pip install git+https://github.com/ograsdijk/CeNTREX-TlF`

# `states`
`states` contains the functions and classes to represent the TlF states:  
`CoupledBasisState` is a class representing a TlF state with coupled quantum numbers, i.e. F, mF, F1, J, I1, I2, Ω, P.  
`UncoupledBasisState` is a class representing a TlF state with uncoupled quantum numbers, i.e. J, mJ, I1, m1, I2, m2, Ω, P.  
Finally `State` is a class representing a collection of states, since in most cases the TlF molecules are in a superposition state.

```Python
from centrex_tlf import states
states.CoupledBasisState(F=1, mF=0, F1 = 1/2, J = 0, I1 = 1/2, I2 = 1/2, Omega = 0, P = 1)
```
or using some of the functions to generate all hyperfine substates in a given J level: 
```Python
from centrex_tlf import states
QN = states.generate_uncoupled_states_ground(Js = [0,1])
```
which returns an array containing the UncoupledBasisStates
```python
array([|X, J = 0, mJ = 0, I₁ = 1/2, m₁ = -1/2, I₂ = 1/2, m₂ = -1/2, P = +, Ω = 0>,
       |X, J = 0, mJ = 0, I₁ = 1/2, m₁ = -1/2, I₂ = 1/2, m₂ = 1/2, P = +, Ω = 0>,
       |X, J = 0, mJ = 0, I₁ = 1/2, m₁ = 1/2, I₂ = 1/2, m₂ = -1/2, P = +, Ω = 0>,
       |X, J = 0, mJ = 0, I₁ = 1/2, m₁ = 1/2, I₂ = 1/2, m₂ = 1/2, P = +, Ω = 0>,
       |X, J = 1, mJ = -1, I₁ = 1/2, m₁ = -1/2, I₂ = 1/2, m₂ = -1/2, P = -, Ω = 0>,
       |X, J = 1, mJ = -1, I₁ = 1/2, m₁ = -1/2, I₂ = 1/2, m₂ = 1/2, P = -, Ω = 0>,
       |X, J = 1, mJ = -1, I₁ = 1/2, m₁ = 1/2, I₂ = 1/2, m₂ = -1/2, P = -, Ω = 0>,
       |X, J = 1, mJ = -1, I₁ = 1/2, m₁ = 1/2, I₂ = 1/2, m₂ = 1/2, P = -, Ω = 0>,
       |X, J = 1, mJ = 0, I₁ = 1/2, m₁ = -1/2, I₂ = 1/2, m₂ = -1/2, P = -, Ω = 0>,
       |X, J = 1, mJ = 0, I₁ = 1/2, m₁ = -1/2, I₂ = 1/2, m₂ = 1/2, P = -, Ω = 0>,
       |X, J = 1, mJ = 0, I₁ = 1/2, m₁ = 1/2, I₂ = 1/2, m₂ = -1/2, P = -, Ω = 0>,
       |X, J = 1, mJ = 0, I₁ = 1/2, m₁ = 1/2, I₂ = 1/2, m₂ = 1/2, P = -, Ω = 0>,
       |X, J = 1, mJ = 1, I₁ = 1/2, m₁ = -1/2, I₂ = 1/2, m₂ = -1/2, P = -, Ω = 0>,
       |X, J = 1, mJ = 1, I₁ = 1/2, m₁ = -1/2, I₂ = 1/2, m₂ = 1/2, P = -, Ω = 0>,
       |X, J = 1, mJ = 1, I₁ = 1/2, m₁ = 1/2, I₂ = 1/2, m₂ = -1/2, P = -, Ω = 0>,
       |X, J = 1, mJ = 1, I₁ = 1/2, m₁ = 1/2, I₂ = 1/2, m₂ = 1/2, P = -, Ω = 0>],
      dtype=object)
```
State objects, which are superpositions of BasisStates are also generated easily:
```Python
superposition = 1*QN[0] + 0.1j*QN[1]
```
which returns
```Python
1.00 x |X, J = 0, mJ = 0, I₁ = 1/2, m₁ = -1/2, I₂ = 1/2, m₂ = -1/2, P = +, Ω = 0>
0.00+0.10j x |X, J = 0, mJ = 0, I₁ = 1/2, m₁ = -1/2, I₂ = 1/2, m₂ = 1/2, P = +, Ω = 0>
```
A subset of `State`, `CoupledBasisStates` can be selected with the `QuantumSelector` as follows:
```Python
QN = states.generate_coupled_states_ground(Js = [0,1])
qn_select = states.QuantumSelector(J = 1, mF = 0, electronic = states.ElectronicState.X)
qn_select.get_indices(QN)
```
which returns all the indices with `J=1` and `mJ=0`:
```python
array([ 4,  6,  9, 13], dtype=int64)
```
# `hamiltonian`
`hamiltonian` contains the functions to generate TlF hamiltonians in the X and B state in either coupled or uncoupled form.  
Generating a ground state X hamiltonian can be accomplished easily using some convenience functions:
```Python
from centrex_tlf import states, hamiltonian

# generate the hyperfine sublevels in J=0 and J=1
QN = states.generate_uncoupled_states_ground(Js = [0,1])

# generate a dictionary with X hamiltonian terms
H = hamiltonian.generate_uncoupled_hamiltonian_X(QN)

# create a function outputting the hamiltonian as a function of E and B
Hfunc = hamiltonian.generate_uncoupled_hamiltonian_X_function(H)
```
All functions generating hamiltonians only require a list or array of TlF states. Generating the hamiltonian only for certain hyperfine sublevels is hence also straightforward. The function `calculate_uncoupled_hamiltonian_X` calculates the hamiltonians from scratch, whereas `generate_uncoupled_hamiltonian_X` pulls the non-zero elements from an sqlite database.

To convert a hamiltonian from one basis to another transformation matrices can be generated or calculated
(`generate_transform_matrix` pulls non-zero matrix elements from an sqlite database, `calculate_transform_matrix` does the full element wise calculation):
```Python
from centrex_tlf import states, hamiltonian

# generate the hyperfine sublevels in J=0 and J=1
QN = states.generate_uncoupled_states_ground(Js = [0,1])
# generate the coupled hyperfine sublevels in J=0 and J=1
QNc = states.generate_coupled_states_ground(Js = [0,1])

# generate a dictionary with X hamiltonian terms
H = hamiltonian.generate_uncoupled_hamiltonian_X(QN)
Hfunc = hamiltonian.generate_uncoupled_hamiltonian_X_function(H)
H0 = Hfunc(E = [0,0,0], B = [0,0,1e-3])

# generate the transformation matrix
transform = hamiltonian.generate_transform_matrix(QN, QNc)

# calculate the transformed matrix
H0c = transform.conj().T@H0@transform
```
This is mostly used for optical bloch simulations where the coupled states representation is more convenient.

## Stark Shift Example
To calculate the energy levels as a function of the electric field the following code can be used, which calculates all energies up to `J=6` but only plots the `|J=2, mJ=0>` hyperfine levels. These are the states focussed by the electrostatic quadrupole lens in the CeNTREX experiment.
![Quadrupole Lens States](quadrupole_lens_states.png)
```Python
import numpy as np
import matplotlib.pyplot as plt

from centrex_tlf import states, hamiltonian

# generate states up to J=6
QN = states.generate_uncoupled_states_ground(Js=np.arange(7))

# generate the X hamiltonian terms
H = hamiltonian.generate_uncoupled_hamiltonian_X(QN)

# create a function outputting the hamiltonian as a function of E and B
Hfunc = hamiltonian.generate_uncoupled_hamiltonian_X_function(H)

# V/cm
Ez = np.linspace(0, 50e3, 101)

# generate the Hamiltonian for (almost) zero field, add a small field to make states
# non-degenerate
Hi = Hfunc(E=[0, 0, 1e-3], B=[0, 0, 1e-3])
E, V = np.linalg.eigh(Hi)

# get the true superposition-states of the system
QN_states = hamiltonian.matrix_to_states(V, QN)

# original eigenvectors used in tracking states as energies change order
V_track = V.copy()

# indices of the J=2, mJ=0 states focused by the lens
indices_J2_mJ0 = [
    idx
    for idx, s in enumerate(QN_states)
    if s.largest.J == 2 and s.largest.mJ == 0
]

indices_J012 = [
    idx for idx, s in enumerate(QN_states) if s.largest.J in [0, 1, 2]
]

# empty array for storing energies
energy = np.empty([Ez.size, len(QN)], dtype=np.complex128)

# iterate over the electric field values
for idx, Ei in enumerate(Ez):
    Hi = Hfunc(E=[0, 0, Ei], B=[0, 0, 1e-3])
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
)
ax.set_xlabel("E [V/cm]")
ax.set_ylabel("Energy [GHz]")
ax.set_title("|J=2, mJ=0> Stark Curve")
ax.grid(True)
plt.show()
```

# `couplings`
Code for generating the CeNTREX TlF couplings. 
Includes code for generating branching ratios, electric dipole coupling elements and coupling fields

## Generating branching ratios
The code below generates branching ratios from `|J'=1, F1'=1/2, mF=0>` to all states in the `J=1` manifold.
```Python
from centrex_tlf import states, couplings

excited_state = states.CoupledBasisState(
    J=1, F1=1 / 2, F=1, mF=0, I1=1 / 2, I2=1 / 2, Omega=1, P=1
)
qn_select = states.QuantumSelector(J=1)
ground_states = [1*s for s in states.generate_coupled_states_X(qn_select)]
br = couplings.calculate_br(1 * excited_state, ground_states)
```

## Generating couplings
The code below generates the coupling fields for the `J=1` manifold to the `J'=1, F1'=1/2, F'=1` manifold. The returned value is a dataclass `CouplingFields` containing the following fields:  
* `ground_main`
* `excited_main`
* `main_coupling`: the electric dipole coupling between `ground_main` and `excited_main`
* `ground_states`: list of all ground states
* `excited_states`: list of all excited states
* `fields`: a list of `CouplingField` dataclasses with the following fields:  
  * `polarization`: polarization vector
  * `field`: coupling field in the `ground_states` + `excited_states` basis

```Python
from centrex_tlf import states, couplings

qn_select = states.QuantumSelector(J=1)
ground_states = states.generate_coupled_states_X(qn_select)

qn_select = states.QuantumSelector(J=1, F1=1 / 2, F=1, P=1, Ω=1)
excited_states = states.generate_coupled_states_B(qn_select)

# the generate_coupling_field_* functions requires lists as inputs, not np.ndarrays
QN = list(1 * np.append(ground_states, excited_states))
ground_states = [1*s for s in  ground_states]
excited_states = [1*s for s in excited_states]

H_rot = np.eye(len(QN), dtype=complex) * np.arange(len(QN))
V_ref = np.eye(len(QN))
pol_vecs = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])]
normalize_pol = True

coupling = couplings.generate_coupling_field_automatic(
    ground_states_approx = ground_states, 
    excited_states_approx = excited_states, 
    QN_basis = QN,
    H_rot = H_rot, 
    QN = QN, 
    V_ref = V_ref, 
    pol_vecs = pol_vecs, 
    normalize_pol = normalize_pol
)
```
