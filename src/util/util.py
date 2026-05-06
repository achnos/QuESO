import numpy as np

def symmetrize_matrix(P_tilde: np.ndarray) -> np.ndarray:
    return (P_tilde + P_tilde.T) / 2.0