"""Tests for the top-level AXIOM agent."""

import numpy as np
import pytest

from axiom.agent import AXIOMAgent


class TestAXIOMAgent:
    def test_initialization(self):
        config = {"seed": 42, "num_actions": 5}
        agent = AXIOMAgent(config)
        assert agent.step_count == 0
        assert agent.slot_latents is None

    def test_random_action_before_observation(self):
        config = {"seed": 42, "num_actions": 5}
        agent = AXIOMAgent(config)
        action = agent.act()
        assert 0 <= action < 5

    def test_bmr_schedule(self):
        config = {"seed": 0, "bmr_interval": 500}
        agent = AXIOMAgent(config)
        agent.step_count = 499
        assert not agent.should_run_bmr()
        agent.step_count = 500
        assert agent.should_run_bmr()
        agent.step_count = 1000
        assert agent.should_run_bmr()
