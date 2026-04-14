"""
Wrapper around the VersesTech/gameworld environment suite.

Provides a uniform interface for all 10 Gameworld 10k games:
Aviate, Bounce, Cross, Drive, Explode, Fruits, Gold, Hunt, Impact, Jump.

Expects gameworld to be installed (see `make setup-gameworld`).
"""

GAME_NAMES = [
    "aviate", "bounce", "cross", "drive", "explode",
    "fruits", "gold", "hunt", "impact", "jump",
]


class GameworldEnv:
    """Thin wrapper providing a consistent step/reset interface for AXIOM."""

    def __init__(self, game_name: str, seed: int = 0):
        if game_name not in GAME_NAMES:
            raise ValueError(f"Unknown game: {game_name}. Choose from {GAME_NAMES}")
        self.game_name = game_name
        self.seed = seed
        self._env = None

    def reset(self) -> dict:
        """Reset environment, return initial observation dict."""
        raise NotImplementedError("Install gameworld first: make setup-gameworld")

    def step(self, action: int) -> tuple:
        """Take action, return (observation, reward, done, info)."""
        raise NotImplementedError

    @property
    def num_actions(self) -> int:
        raise NotImplementedError

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """(H, W, C) shape of RGB observations."""
        raise NotImplementedError
