"""
DreamerV3 baseline wrapper.

DreamerV3 is a world-model-based agent with strong performance on games and
control tasks from pixel inputs.

Reference: Hafner et al., "Mastering Diverse Domains through World Models" (2023).
"""


class DreamerV3Agent:
    """Wrapper around the DreamerV3 implementation for Gameworld evaluation."""

    def __init__(self, config: dict):
        self.config = config
        raise NotImplementedError(
            "DreamerV3 wrapper not yet implemented. "
            "See https://github.com/danijar/dreamerv3"
        )
