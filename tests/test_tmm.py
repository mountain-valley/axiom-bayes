"""Tests for the Transition Mixture Model."""

import numpy as np
import pytest

from axiom.models.tmm import TransitionMixtureModel


class TestTransitionMixtureModel:
    def test_initialization(self):
        tmm = TransitionMixtureModel(max_modes=16, state_dim=8)
        assert tmm.max_modes == 16
        assert tmm.state_dim == 8
        assert tmm.num_active_modes == 0

    # TODO: Add tests as the tMM is implemented
    # - test linear prediction x_next = D @ x_prev + b
    # - test mode expansion for a novel trajectory
    # - test shared modes across multiple slots
