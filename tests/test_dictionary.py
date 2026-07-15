"""Stage 3 gate: the learning loop actually descends on a toy problem.

Reconstruction error should decrease monotonically (allowing for batch noise)
over iterations on a fixed synthetic problem, and dictionary columns should stay
unit-norm after the renormalization step.
"""

import pytest


@pytest.mark.skip(reason="Stage 3 not implemented")
def test_reconstruction_error_decreases():
    raise NotImplementedError


@pytest.mark.skip(reason="Stage 3 not implemented")
def test_dictionary_columns_stay_unit_norm():
    raise NotImplementedError
