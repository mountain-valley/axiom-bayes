"""Tests for variational inference and structure learning."""

import numpy as np
import pytest

from axiom.inference.structure_learning import (
    compute_threshold,
    should_expand,
    assign_or_expand,
)


class TestStructureLearning:
    def test_compute_threshold(self):
        tau = compute_threshold(prior_predictive_log_density=-5.0, alpha=1.0)
        assert tau == pytest.approx(-5.0 + np.log(1.0))

    def test_should_expand_when_below_threshold(self):
        scores = np.array([-10.0, -8.0, -7.0])
        assert should_expand(scores, threshold=-5.0, num_active=3, max_components=10)

    def test_should_not_expand_when_above_threshold(self):
        scores = np.array([-3.0, -8.0, -7.0])
        assert not should_expand(scores, threshold=-5.0, num_active=3, max_components=10)

    def test_should_not_expand_when_at_capacity(self):
        scores = np.array([-10.0])
        assert not should_expand(scores, threshold=-5.0, num_active=10, max_components=10)

    def test_assign_or_expand_existing(self):
        scores = np.array([-2.0, -5.0, -8.0])
        idx, is_new = assign_or_expand(
            data_point=np.zeros(3), component_scores=scores,
            threshold=-5.0, num_active=3, max_components=10,
        )
        assert idx == 0
        assert not is_new

    def test_assign_or_expand_new(self):
        scores = np.array([-10.0, -8.0])
        idx, is_new = assign_or_expand(
            data_point=np.zeros(3), component_scores=scores,
            threshold=-5.0, num_active=2, max_components=10,
        )
        assert idx == 2
        assert is_new
