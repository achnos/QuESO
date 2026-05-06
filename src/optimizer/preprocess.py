"""
Preprocessing pipeline for QuESO.

Reduces the full problem (P, W, F, A) to a smaller active subproblem
by eliminating employees with no attendance and those with fixed seat
assignments, and modulates the affinity matrix by shared attendance days.
Fixed-seat employees are eliminated as decision variables but their affinity
with active employees is retained as a linear bias for the QUBO constructor.
"""
from dataclasses import dataclass
import numpy as np

import util


@dataclass
class PreprocessingResult:
    P_tilde: np.ndarray        # shape (n_active, n_active) - modulated, symmetrized affinity
    W: np.ndarray              # shape (n_active, T) - attendance for active employees
    F: np.ndarray              # shape (n_active, S) - fixed seat matrix for active employees
    A: np.ndarray              # shape (S+1, S+1) - seat adjacency, passed through unchanged
    P_fixed_active: np.ndarray # shape (n_fixed, n_active) - affinity between fixed and active
    fixed_seats_map: list[int]     # seat indices of fixed employees, aligned with P_fixed_active rows
    employee_index_map: dict   # keys: pre_fixed, no_attendance, active
    n_active: int              # number of active employees remaining


def _get_idxs_no_attendance_ns(W: np.ndarray) -> list[int]:
    """Return original indices of employees with no attendance during T."""
    return [int(n) for n in np.where(W.sum(axis=1) == 0)[0]]


def _get_fixed_seatings_map(F: np.ndarray) -> dict[int, int]:
    """Return mapping of original employee index -> seat index for fixed assignments."""
    fixed_seatings = {}
    for n in np.where(F.sum(axis=1) > 0)[0]:
        seat = int(np.where(F[n] == 1)[0][0])
        fixed_seatings[int(n)] = seat
    return fixed_seatings


def _modulate_affinity(P: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Modulate affinity matrix by shared attendance days and symmetrize.

    P_tilde_nm = P_nm * sum_t(W_nt * W_mt), then symmetrized as (P + P.T) / 2.
    """
    shared_days = W @ W.T  # shape (n_active, n_active)
    P_tilde = P * shared_days
    return util.symmetrize_matrix(P_tilde)


def _get_P_contribution_from_fixed_seatings(P: np.ndarray, T,
                                            W: np.ndarray,
                                            W_active: np.ndarray, idxs_active_ns: list[int],
                                            idxs_fixed_seatings_ns: list[int]) -> np.ndarray:
    W_fixed = W[np.ix_(idxs_fixed_seatings_ns, list(range(T)))]  # shape (n_fixed, T)
    shared_days_fixed_active = W_fixed @ W_active.T  # shape (n_fixed, n_active)
    P_fixed_active = P[np.ix_(idxs_fixed_seatings_ns, idxs_active_ns)] * shared_days_fixed_active
    return P_fixed_active


def _get_employee_idx_map(fixed_seatings_dict: dict[int, int], idxs_active_ns: list[int],
                          idxs_no_attendance_ns: list[int]) -> dict[str, list[int] | dict[int, int]]:
    return {
        "no_attendance": idxs_no_attendance_ns,
        "pre_fixed": fixed_seatings_dict,
        "active": {idxs_ns_reduced_matrix: idxs_ns_original_matrix for idxs_ns_reduced_matrix, idxs_ns_original_matrix in enumerate(idxs_active_ns)},
    }


def _get_idxs_of_active(N, no_attendance: list[int], pre_fixed: dict[int, int]) -> list[int]:
    eliminated = set(no_attendance) | set(pre_fixed.keys())
    active_original = [n for n in range(N) if n not in eliminated]
    return active_original


def _get_matrices_contributions_from_active_employees(S, T, active_original: list[int], F: np.ndarray, P: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    W_active = W[np.ix_(active_original, list(range(T)))]
    F_active = F[np.ix_(active_original, list(range(S)))]
    P_active = P[np.ix_(active_original, active_original)]
    return F_active, P_active, W_active


def preprocess(
    P: np.ndarray,
    W: np.ndarray,
    F: np.ndarray,
    A: np.ndarray,
) -> PreprocessingResult:
    """
    Run the full preprocessing pipeline.

    Parameters
    ----------
    P : np.ndarray, shape (N, N)
        Raw pairwise affinity matrix.
    W : np.ndarray, shape (N, T)
        Attendance matrix. W[n, t] = 1 if employee n is onsite on day t.
    F : np.ndarray, shape (N, S)
        Fixed assignment matrix. F[n, s] = 1 if employee n must sit at seat s.
    A : np.ndarray, shape (S+1, S+1)
        Seat adjacency matrix, upper triangular, zero diagonal.

    Returns
    -------
    PreprocessingResult
    """
    N = P.shape[0]
    T = W.shape[1]
    S = F.shape[1]

    idxs_no_attendance_ns = _get_idxs_no_attendance_ns(W)
    fixed_seatings_dict = _get_fixed_seatings_map(F)
    idxs_active_ns = _get_idxs_of_active(N, idxs_no_attendance_ns, fixed_seatings_dict)
    employee_index_map = _get_employee_idx_map(fixed_seatings_dict, idxs_active_ns, idxs_no_attendance_ns)

    F_active, P_active, W_active = _get_matrices_contributions_from_active_employees(S, T, idxs_active_ns, F, P, W)

    # Uses P_tilde_{n* n} = P_{n* n} * sum_t W_{n* t} W_{nt}, matching the paper's Delta Q term
    P_fixed_active = _get_P_contribution_from_fixed_seatings(P, T, W, W_active, idxs_active_ns, list(fixed_seatings_dict.keys()))
    P_tilde = _modulate_affinity(P_active, W_active)

    return PreprocessingResult(
        P_tilde=P_tilde,
        W=W_active,
        F=F_active,
        A=A,
        P_fixed_active=P_fixed_active,
        fixed_seats_map=(list(fixed_seatings_dict.values())),
        employee_index_map=employee_index_map,
        n_active=len(idxs_active_ns),
    )
