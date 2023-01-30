from dataclasses import dataclass

__all__ = ["TlFNuclearSpins"]


@dataclass
class TlFNuclearSpins:
    I_F: float = 1 / 2
    I_Tl: float = 1 / 2
