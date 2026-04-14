"""
Transition Mixture Model (tMM).

Models per-slot dynamics as a switching linear dynamical system (SLDS). Each of L
linear modes captures a distinct rigid motion pattern (e.g., "ball falling",
"paddle left"). Modes are shared across all K slots. Fixed covariance (2I).

Reference: AXIOM paper Section 2, Equation 5.
"""

import numpy as np


class TransitionMixtureModel:
    """Piecewise-linear dynamics model with shared motion modes across slots."""

    def __init__(self, max_modes: int, state_dim: int, alpha_tmm: float = 1.0):
        self.max_modes = max_modes
        self.state_dim = state_dim
        self.alpha_tmm = alpha_tmm
        self.num_active_modes = 0
        self.linear_params: list[dict] = []

    def predict(
        self, x_prev: np.ndarray, switch_state: int
    ) -> np.ndarray:
        """
        Predict next state given previous state and active linear mode.

        x_next = D_l @ x_prev + b_l

        Args:
            x_prev: (state_dim,) previous slot state.
            switch_state: index of the active linear mode.

        Returns:
            (state_dim,) predicted next state.
        """
        raise NotImplementedError

    def update_params(
        self, x_prev: np.ndarray, x_curr: np.ndarray, switch_state: int
    ):
        """M-step: update linear parameters D_l, b_l for the active mode."""
        raise NotImplementedError

    def expand_if_needed(self, x_prev: np.ndarray, x_curr: np.ndarray):
        """Grow a new linear mode if the transition is not explained by existing modes."""
        raise NotImplementedError
