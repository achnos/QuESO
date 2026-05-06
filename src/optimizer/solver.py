"""
Solver interface for QuESO.

Provides a thin, problem-agnostic wrapper around dimod samplers.
The default sampler is neal.SimulatedAnnealingSampler, but any
dimod-compatible sampler can be substituted at the call site.

solve()            -- single solve, returns raw SampleSet
solve_with_retry() -- retries until a feasible sample is found or
                      max_retries is exhausted, feasibility defined
                      by the caller via an injected callable
"""

from collections.abc import Callable

import dimod
import neal


def solve(
    bqm: dimod.BinaryQuadraticModel,
    sampler: dimod.Sampler | None = None,
    **sampler_kwargs,
) -> dimod.SampleSet:
    """
    Sample from a BinaryQuadraticModel using the provided sampler.

    Parameters
    ----------
    bqm : dimod.BinaryQuadraticModel
        The QUBO problem to solve.
    sampler : dimod.Sampler or None
        Any dimod-compatible sampler. Defaults to
        neal.SimulatedAnnealingSampler if None.
    **sampler_kwargs
        Keyword arguments forwarded to sampler.sample().

    Returns
    -------
    dimod.SampleSet
        Raw sample set returned by the sampler, sorted by energy.
    """
    if sampler is None:
        sampler = neal.SimulatedAnnealingSampler()

    sample_set = sampler.sample(bqm, **sampler_kwargs)
    return sample_set


def solve_with_retry(
    bqm: dimod.BinaryQuadraticModel,
    is_valid: Callable[[dimod.SampleSet], bool],
    sampler: dimod.Sampler | None = None,
    max_retries: int = 3,
    **sampler_kwargs,
) -> tuple[dimod.SampleSet, bool]:
    """
    Repeatedly solve until a feasible sample is found or retries
    are exhausted. Feasibility is defined by the caller.

    Parameters
    ----------
    bqm : dimod.BinaryQuadraticModel
        The QUBO problem to solve.
    is_valid : Callable[[dimod.SampleSet], bool]
        A function that takes a SampleSet and returns True if the
        best sample is considered feasible.
    sampler : dimod.Sampler or None
        Any dimod-compatible sampler. Defaults to
        neal.SimulatedAnnealingSampler if None.
    max_retries : int
        Maximum number of solve attempts. Default 3.
    **sampler_kwargs
        Keyword arguments forwarded to sampler.sample().

    Returns
    -------
    sample_set : dimod.SampleSet
        The last SampleSet obtained, whether feasible or not.
    feasible : bool
        True if a feasible sample was found within max_retries attempts.
    """
    sample_set = None
    for attempt in range(max_retries):
        sample_set = solve(bqm, sampler=sampler, **sampler_kwargs)
        if is_valid(sample_set):
            return sample_set, True

    return sample_set, False