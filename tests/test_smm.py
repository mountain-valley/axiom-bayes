"""Tests for the Slot Mixture Model."""

import numpy as np
import pytest

from axiom.models.smm import SlotMixtureModel


class TestSlotMixtureModel:
    def test_initialization(self):
        smm = SlotMixtureModel(max_slots=8, alpha_smm=1.0)
        assert smm.max_slots == 8
        assert smm.num_active_slots == 0

    # TODO: Add tests as the sMM is implemented
    # - test pixel token assignment on a synthetic image with known objects
    # - test slot expansion when a new object appears
    # - test posterior-predictive density computation
