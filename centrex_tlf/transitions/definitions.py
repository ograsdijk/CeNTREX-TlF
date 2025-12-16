"""Predefined optical transition constants for `centrex_tlf.transitions`.

This module holds the manual OpticalTransition constants and alias names
(e.g., `R0_F1_3o2_F2`, `R0F2`) that were moved out of `transition.py` to keep
that file focused on the dataclass/type definitions.

The module imports the `OpticalTransition` and `OpticalTransitionType` from
`.transition` (relative import). `transition.py` imports this module after
its classes are defined to re-export these symbols.
"""

from .transition import OpticalTransition, OpticalTransitionType

__all__ = [
    "R0_F1_1o2_F0",
    "R0_F1_1o2_F1",
    "R0_F1_3o2_F1",
    "R0_F1_3o2_F2",
    "Q1_F1_1o2_F0",
    "Q1_F1_1o2_F1",
    "Q1_F1_3o2_F1",
    "Q1_F1_3o2_F2",
    "R1_F1_3o2_F1",
    "R1_F1_3o2_F2",
    "R1_F1_5o2_F2",
    "R1_F1_5o2_F3",
    "P2_F1_1o2_F0",
    "P2_F1_1o2_F1",
    "P2_F1_3o2_F1",
    "P2_F1_3o2_F2",
    "Q2_F1_3o2_F1",
    "Q2_F1_3o2_F2",
    "Q2_F1_5o2_F2",
    "Q2_F1_5o2_F3",
    "R2_F1_5o2_F2",
    "R2_F1_5o2_F3",
    "R2_F1_7o2_F3",
    "R2_F1_7o2_F4",
    "P3_F1_3o2_F1",
    "P3_F1_3o2_F2",
    "P3_F1_5o2_F2",
    "P3_F1_5o2_F3",
    "Q3_F1_5o2_F2",
    "Q3_F1_5o2_F3",
    "Q3_F1_7o2_F3",
    "Q3_F1_7o2_F4",
    "R3_F1_7o2_F3",
    "R3_F1_7o2_F4",
    "R3_F1_9o2_F4",
    "R3_F1_9o2_F5",
    "P4_F1_5o2_F2",
    "P4_F1_5o2_F3",
    "P4_F1_7o2_F3",
    "P4_F1_7o2_F4",
    "Q4_F1_7o2_F3",
    "Q4_F1_7o2_F4",
    "Q4_F1_9o2_F4",
    "Q4_F1_9o2_F5",
    "R4_F1_9o2_F4",
    "R4_F1_9o2_F5",
    "R4_F1_11o2_F5",
    "R4_F1_11o2_F6",
    # Aliases
    "R0F2",
    "R1F3",
    "R2F4",
    "R3F5",
    "R4F6",
    "Q1F2",
    "Q2F3",
    "Q3F4",
    "Q4F5",
    "P2F2",
    "P3F3",
    "P4F4",
]

# Transition constants (manual list)
R0_F1_1o2_F0 = OpticalTransition(OpticalTransitionType.R, 0, 1/2, 0)
R0_F1_1o2_F1 = OpticalTransition(OpticalTransitionType.R, 0, 1/2, 1)
R0_F1_3o2_F1 = OpticalTransition(OpticalTransitionType.R, 0, 3/2, 1)
R0_F1_3o2_F2 = OpticalTransition(OpticalTransitionType.R, 0, 3/2, 2)

Q1_F1_1o2_F0 = OpticalTransition(OpticalTransitionType.Q, 1, 1/2, 0)
Q1_F1_1o2_F1 = OpticalTransition(OpticalTransitionType.Q, 1, 1/2, 1)
Q1_F1_3o2_F1 = OpticalTransition(OpticalTransitionType.Q, 1, 3/2, 1)
Q1_F1_3o2_F2 = OpticalTransition(OpticalTransitionType.Q, 1, 3/2, 2)
R1_F1_3o2_F1 = OpticalTransition(OpticalTransitionType.R, 1, 3/2, 1)
R1_F1_3o2_F2 = OpticalTransition(OpticalTransitionType.R, 1, 3/2, 2)
R1_F1_5o2_F2 = OpticalTransition(OpticalTransitionType.R, 1, 5/2, 2)
R1_F1_5o2_F3 = OpticalTransition(OpticalTransitionType.R, 1, 5/2, 3)

P2_F1_1o2_F0 = OpticalTransition(OpticalTransitionType.P, 2, 1/2, 0)
P2_F1_1o2_F1 = OpticalTransition(OpticalTransitionType.P, 2, 1/2, 1)
P2_F1_3o2_F1 = OpticalTransition(OpticalTransitionType.P, 2, 3/2, 1)
P2_F1_3o2_F2 = OpticalTransition(OpticalTransitionType.P, 2, 3/2, 2)
Q2_F1_3o2_F1 = OpticalTransition(OpticalTransitionType.Q, 2, 3/2, 1)
Q2_F1_3o2_F2 = OpticalTransition(OpticalTransitionType.Q, 2, 3/2, 2)
Q2_F1_5o2_F2 = OpticalTransition(OpticalTransitionType.Q, 2, 5/2, 2)
Q2_F1_5o2_F3 = OpticalTransition(OpticalTransitionType.Q, 2, 5/2, 3)
R2_F1_5o2_F2 = OpticalTransition(OpticalTransitionType.R, 2, 5/2, 2)
R2_F1_5o2_F3 = OpticalTransition(OpticalTransitionType.R, 2, 5/2, 3)
R2_F1_7o2_F3 = OpticalTransition(OpticalTransitionType.R, 2, 7/2, 3)
R2_F1_7o2_F4 = OpticalTransition(OpticalTransitionType.R, 2, 7/2, 4)

P3_F1_3o2_F1 = OpticalTransition(OpticalTransitionType.P, 3, 3/2, 1)
P3_F1_3o2_F2 = OpticalTransition(OpticalTransitionType.P, 3, 3/2, 2)
P3_F1_5o2_F2 = OpticalTransition(OpticalTransitionType.P, 3, 5/2, 2)
P3_F1_5o2_F3 = OpticalTransition(OpticalTransitionType.P, 3, 5/2, 3)
Q3_F1_5o2_F2 = OpticalTransition(OpticalTransitionType.Q, 3, 5/2, 2)
Q3_F1_5o2_F3 = OpticalTransition(OpticalTransitionType.Q, 3, 5/2, 3)
Q3_F1_7o2_F3 = OpticalTransition(OpticalTransitionType.Q, 3, 7/2, 3)
Q3_F1_7o2_F4 = OpticalTransition(OpticalTransitionType.Q, 3, 7/2, 4)
R3_F1_7o2_F3 = OpticalTransition(OpticalTransitionType.R, 3, 7/2, 3)
R3_F1_7o2_F4 = OpticalTransition(OpticalTransitionType.R, 3, 7/2, 4)
R3_F1_9o2_F4 = OpticalTransition(OpticalTransitionType.R, 3, 9/2, 4)
R3_F1_9o2_F5 = OpticalTransition(OpticalTransitionType.R, 3, 9/2, 5)

P4_F1_5o2_F2 = OpticalTransition(OpticalTransitionType.P, 4, 5/2, 2)
P4_F1_5o2_F3 = OpticalTransition(OpticalTransitionType.P, 4, 5/2, 3)
P4_F1_7o2_F3 = OpticalTransition(OpticalTransitionType.P, 4, 7/2, 3)
P4_F1_7o2_F4 = OpticalTransition(OpticalTransitionType.P, 4, 7/2, 4)
Q4_F1_7o2_F3 = OpticalTransition(OpticalTransitionType.Q, 4, 7/2, 3)
Q4_F1_7o2_F4 = OpticalTransition(OpticalTransitionType.Q, 4, 7/2, 4)
Q4_F1_9o2_F4 = OpticalTransition(OpticalTransitionType.Q, 4, 9/2, 4)
Q4_F1_9o2_F5 = OpticalTransition(OpticalTransitionType.Q, 4, 9/2, 5)
R4_F1_9o2_F4 = OpticalTransition(OpticalTransitionType.R, 4, 9/2, 4)
R4_F1_9o2_F5 = OpticalTransition(OpticalTransitionType.R, 4, 9/2, 5)
R4_F1_11o2_F5 = OpticalTransition(OpticalTransitionType.R, 4, 11/2, 5)
R4_F1_11o2_F6 = OpticalTransition(OpticalTransitionType.R, 4, 11/2, 6)

# Aliases: R-branch, one symbol per ground J selecting the largest F' available
# e.g. R0F2 is the R-branch transition from J=0 to F'=2 (largest F' for J=0)
R0F2 = R0_F1_3o2_F2
R1F3 = R1_F1_5o2_F3
R2F4 = R2_F1_7o2_F4
R3F5 = R3_F1_9o2_F5
R4F6 = R4_F1_11o2_F6

# Aliases: Q-branch, one symbol per ground J selecting the largest F' available
# e.g. Q1F2 is the Q-branch transition from J=1 to F'=2 (largest F' for J=1)
Q1F2 = Q1_F1_3o2_F2
Q2F3 = Q2_F1_5o2_F3
Q3F4 = Q3_F1_7o2_F4
Q4F5 = Q4_F1_9o2_F5

# Aliases: P-branch, one symbol per ground J selecting the largest F' available
# e.g. P2F2 is the P-branch transition from J=2 to F'=2 (largest F' for J=2)
P2F2 = P2_F1_3o2_F2
P3F3 = P3_F1_5o2_F3
P4F4 = P4_F1_7o2_F4
