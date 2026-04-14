"""Tests for the Identity Mixture Model."""

import numpy as np
import pytest

from axiom.models.imm import IdentityMixtureModel


class TestIdentityMixtureModel:
    def test_initialization(self):
        imm = IdentityMixtureModel(max_types=5, alpha_imm=1.0)
        assert imm.max_types == 5
        assert imm.num_active_types == 0

    # TODO: Add tests as the iMM is implemented
    # - test identity assignment with distinct color/shape clusters
    # - test NIW posterior update correctness
    # - test type expansion
