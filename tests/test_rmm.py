"""Tests for the Recurrent Mixture Model."""

import numpy as np
import pytest

from axiom.models.rmm import RecurrentMixtureModel


class TestRecurrentMixtureModel:
    def test_initialization(self):
        rmm = RecurrentMixtureModel(max_components=64, alpha_rmm=1.0)
        assert rmm.max_components == 64
        assert rmm.num_active_components == 0

    # TODO: Add tests as the rMM is implemented
    # - test switch state inference from known interaction features
    # - test component expansion
    # - test Gaussian × Categorical likelihood computation
