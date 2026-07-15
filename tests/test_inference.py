"""Stage 2 gate: both owned solvers agree with sklearn LASSO on a planted problem.

This is the hard gate from CLAUDE.md -- no sparse solver touches real patches
until it passes. Build a planted-sparse-code problem (random dictionary + known
sparse codes -> observations), then check that LCA and ISTA/FISTA both (a)
recover the true support and (b) agree with sklearn's Lasso within tolerance on
the identical (Phi, x).

Tolerance is a decision to be recorded at Stage 2, not guessed here.
"""

import pytest


@pytest.mark.skip(reason="Stage 2 not implemented")
def test_lca_recovers_planted_support():
    raise NotImplementedError


@pytest.mark.skip(reason="Stage 2 not implemented")
def test_ista_recovers_planted_support():
    raise NotImplementedError


@pytest.mark.skip(reason="Stage 2 not implemented")
def test_solvers_agree_with_sklearn_lasso():
    raise NotImplementedError
