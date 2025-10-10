"""Hamiltonian operators for TlF B electronic state in coupled Omega basis.

This subpackage provides all Hamiltonian operator implementations for the B(¹)²Π₁/₂
electronic state of TlF using the coupled basis |J,F₁,F,mF,I₁,I₂,Ω⟩ where Omega
(projection of electronic angular momentum) is a good quantum number.

The B state is a ²Π₁/₂ state with significant lambda-doubling and spin-orbit effects.
This basis is particularly convenient for describing states with definite parity and
well-defined Omega projection.

Modules:
    rotational: Rotational energy (B·J² - D·J⁴ + H·J⁶) with centrifugal distortion
    ld: Lambda-doubling interactions (q-term and nuclear spin-rotation c'₁)
    mhf: Magnetic hyperfine interactions for Tl and F nuclei
    nsr: Nuclear spin-rotation interaction for Tl nucleus (c-term)
    stark: Electric dipole interactions with external electric fields
    zeeman: Magnetic dipole interactions with external magnetic fields

The operators are implemented with LRU caching for performance and use Wigner 3-j
and 6-j symbols for angular momentum coupling calculations.

Example:
    >>> from centrex_tlf.hamiltonian import B_coupled_Omega
    >>> from centrex_tlf.states import CoupledBasisState
    >>> from centrex_tlf.constants import BConstants
    >>>
    >>> state = CoupledBasisState(F=1, mF=0, F1=0.5, J=1, I1=0.5, I2=0.5, Omega=1)
    >>> constants = BConstants()
    >>>
    >>> # Calculate rotational energy
    >>> E_rot = B_coupled_Omega.rotational.Hrot(state, constants)
    >>>
    >>> # Calculate Stark shift
    >>> H_stark = B_coupled_Omega.stark.HSz(state, constants)
"""

from . import ld, mhf, nsr, rotational, stark, zeeman

__all__ = ["ld", "mhf", "nsr", "rotational", "stark", "zeeman"]
