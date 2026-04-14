"""
BBF (Bigger, Better, Faster) baseline wrapper.

BBF builds on SR-SPR and is one of the most sample-efficient model-free approaches.
This module provides a thin wrapper to run the published BBF implementation
against Gameworld environments.

Reference: Schwarzer et al., "Bigger, Better, Faster" (2023).
"""


class BBFAgent:
    """Wrapper around the BBF implementation for Gameworld evaluation."""

    def __init__(self, config: dict):
        self.config = config
        raise NotImplementedError(
            "BBF wrapper not yet implemented. "
            "See https://github.com/google-research/google-research/tree/master/bigger_better_faster"
        )
