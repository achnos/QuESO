"""
QUBO construction for QuESO.

Builds a BinaryQuadraticModel from a PreprocessingResult by assembling
the QUBO matrix elements as derived in the QuESO paper, Table 1:

    Diagonal (n=m, s=s'):          -lambda_1 + Delta_Q_uu
    Same employee, diff seats:      +lambda_1 - P_tilde_nn * A_ss'
    Diff employees, same seat:      +lambda_2
    Diff employees, diff seats:     -P_tilde_nm * A_ss'

The flattening map is mu(n, s) = n * (S + 1) + s, giving a QUBO of
size N_active * (S + 1) x N_active * (S + 1).

A is assumed upper triangular with zero diagonal, consistent with the
paper's convention. The BQM is constructed in BINARY vartype.
"""
from typing import Any

import dimod
import numpy as np

from .preprocess import PreprocessingResult

DEFAULT_PENALTY_MULTIPLIER = 2.0


def mu(n: int, s: int, S: int) -> int:
    """Flattening map"""
    return n * (S + 1) + s


def _auto_penalty(P_tilde: np.ndarray, S: int, multiplier: float) -> float:
    """
    Compute penalty weight from the rule of thumb:
        lambda = multiplier * max_nm |P_tilde_nm| * S

    If P_tilde is all zeros (no affinities), fall back to 1.0 * multiplier
    to ensure constraints are still enforced.
    """
    max_affinity = np.abs(P_tilde).max()
    base = max_affinity * S if max_affinity > 0 else 1.0
    return multiplier * base


def _get_penalty(P_tilde, S: Any, penalty_input: float | None, penalty_multiplier: float) -> float:
    return penalty_input if penalty_input is not None else _auto_penalty(P_tilde, S, penalty_multiplier)


def _get_delta_Q(P_fixed_active: np.ndarray, A: np.ndarray, N_active: int, S: Any, fixed_seats: list[int]) -> np.ndarray:
    """
    Linear biases from fixed-seat employees (Delta Q_uu)
    Shape: (N_active, S+1)
    """
    delta_Q = np.zeros((N_active, (S + 1)))
    for n_star, s_star in enumerate(fixed_seats):
        for m in range(N_active):
            for v in range(S + 1):
                delta_Q[m, v] += (
                        -P_fixed_active[n_star, m] * A[s_star, v]
                )
    return delta_Q




def build_qubo(
    result: PreprocessingResult,
    lambda_1: float | None = None,
    lambda_2: float | None = None,
    penalty_multiplier: float = DEFAULT_PENALTY_MULTIPLIER,
) -> dimod.BinaryQuadraticModel:
    """
    Construct a BinaryQuadraticModel from a PreprocessingResult.

    Parameters
    ----------
    result : PreprocessingResult
        Output of the preprocessing pipeline.
    lambda_1 : float or None
        Penalty weight for the row (one-hot) constraint. If None,
        auto-computed as penalty_multiplier * max|P_tilde| * S.
    lambda_2 : float or None
        Penalty weight for the column (no seat sharing) constraint.
        If None, auto-computed identically to lambda_1.
    penalty_multiplier : float
        Safety multiplier alpha for auto-computed penalties. Default 2.0.

    Returns
    -------
    dimod.BinaryQuadraticModel
        BQM in BINARY vartype encoding the full QUBO problem.
    """
    S = result.A.shape[0] - 1

    lambda_1 = _get_penalty(result.P_tilde, S, lambda_1, penalty_multiplier)
    lambda_2 = _get_penalty(result.P_tilde, S, lambda_2, penalty_multiplier)

    delta_Q = _get_delta_Q(result.P_fixed_active, result.A, result.n_active, S, result.fixed_seats_map)

    bqm = dimod.BinaryQuadraticModel(vartype="BINARY")
    bqm = _construct_bqm(bqm, result.P_tilde, result.A, delta_Q, result.n_active, S, lambda_1, lambda_2)

    return bqm


def _construct_bqm(bqm: dimod.BinaryQuadraticModel,
                   P_tilde: np.ndarray,
                   A: np.ndarray,
                   delta_Q: np.ndarray,
                   N_active: int,
                   S: int,
                   lambda_1: float,
                   lambda_2: float
                   ) -> dimod.BinaryQuadraticModel:
    bqm = _initialize_all_variables(bqm, N_active, S)
    bqm = _set_diagonal_variables(bqm, delta_Q, N_active, S, lambda_1)
    bqm = _set_off_diagonal_variables(bqm, P_tilde, A, N_active, S, lambda_1, lambda_2)
    return bqm


def _initialize_all_variables(bqm: dimod.BinaryQuadraticModel,
                              N_active: int,
                              S: int
                              ) -> dimod.BinaryQuadraticModel:
    # Add all variables explicitly so isolated variables are present
    for m in range(N_active):
        for s in range(S + 1):
            bqm.add_variable(mu(m, s, S), 0.0)
    return bqm


def _set_diagonal_variables(bqm: dimod.BinaryQuadraticModel,
                            delta_Q: np.ndarray,
                            N_active: int,
                            S: int,
                            lambda_1: float
                            ) -> dimod.BinaryQuadraticModel:
    # Diagonal elements: -lambda_1 + Delta_Q_uu
    for m in range(N_active):
        for s in range(S + 1):
            bias = -lambda_1 + delta_Q[m, s]
            bqm.add_variable(mu(m, s, S), bias)
    return bqm


def _set_off_diagonal_variables(bqm: dimod.BinaryQuadraticModel,
                                P_tilde: np.ndarray,
                                A: np.ndarray,
                                N_active: int,
                                S: int,
                                lambda_1: float,
                                lambda_2: float
                                ) -> dimod.BinaryQuadraticModel:
    # Off-diagonal elements — iterate over all unique pairs (u, v) with u < v
    for n in range(N_active):
        for m in range(N_active):
            for s in range(S + 1):
                for s_prime in range(S + 1):
                    u = mu(n, s, S)
                    v = mu(m, s_prime, S)

                    if _is_upper_triangle_of_matrix(u, v):
                        continue

                    contribution = 0.0
                    if _is_same_employee_different_seats(n, m, s, s_prime):
                        contribution += lambda_1 - P_tilde[n, n] * A[s, s_prime]
                    elif _is_different_employees_same_real_seat(n, m, s, s_prime, S):
                        contribution += lambda_2
                    elif _is_different_employees_different_seats(n, m, s, s_prime):
                        contribution += -P_tilde[n, m] * A[s, s_prime]

                    if contribution != 0.0:
                        bqm.add_interaction(u, v, contribution)
    return bqm


def _is_upper_triangle_of_matrix(u: int, v: int) -> bool:
    return u >= v


def _is_different_employees_different_seats(n: int, m: int, s: int, s_prime: int) -> bool:
    return n != m and s != s_prime


def _is_different_employees_same_real_seat(n: int, m: int, s: int, s_prime: int, S: int) -> bool:
    return n != m and s == s_prime and s < S


def _is_same_employee_different_seats(n: int, m: int, s: int, s_prime: int) -> bool:
    return n == m and s != s_prime
