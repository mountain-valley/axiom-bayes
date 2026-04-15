"""Verify AXIOM and Gameworld are installed correctly (Roadmap Task 0)."""

from pathlib import Path
import sys

import numpy as np


class TestAXIOMInstall:
    def test_import_axiom(self):
        import axiom  # noqa: F401

    def test_import_axiom_defaults(self):
        axiom_root = Path(__file__).resolve().parents[1] / "vendor" / "axiom"
        sys.path.insert(0, str(axiom_root))
        import defaults

        args = defaults.parse_args(["--game", "Explode", "--num_steps", "10"])
        assert args.game == "Explode"
        assert args.num_steps == 10


class TestGameworldInstall:
    def test_import_gameworld(self):
        import gameworld  # noqa: F401

    def test_create_env(self):
        import gameworld.envs  # noqa: F401  (registers gymnasium IDs)
        import gymnasium

        env = gymnasium.make("Gameworld-Explode-v0")
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.ndim == 3

        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        assert obs2 is not None
        assert obs2.shape == obs.shape
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()
