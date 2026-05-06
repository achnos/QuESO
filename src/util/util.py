import numpy as np

def symmetrize_matrix(P_tilde: np.ndarray) -> np.ndarray:
    return (P_tilde + P_tilde.T) / 2.0


def mu(n: int, s: int, S: int) -> int:
    """Flattening map"""
    return n * (S + 1) + s