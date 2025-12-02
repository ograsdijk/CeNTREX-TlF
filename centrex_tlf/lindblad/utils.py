import numpy as np

__all__: list[str] = []


def has_off_diagonal_elements(arr: np.ndarray, tol: float = 0.0) -> bool:
    """
    Check if a square NumPy array has any nonzero off-diagonal elements.

    Parameters:
        arr (np.ndarray): A 2D square NumPy array.
        tol (float): Optional tolerance. Any absolute value larger than this is considered nonzero.

    Returns:
        bool: True if any off-diagonal element has absolute value > tol, False otherwise.
    """
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Array must be square")

    off_diag = arr.copy()
    np.fill_diagonal(off_diag, 0)

    return bool(np.any(np.abs(off_diag) > tol))
