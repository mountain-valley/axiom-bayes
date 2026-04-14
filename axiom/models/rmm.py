"""
Recurrent Mixture Model (rMM).

Models the joint distribution of continuous slot features, discrete slot features
(identity, tMM switch state), action, and reward as a mixture model. This is the
module that captures sparse object-object interactions — conditioning tMM switch
states on multi-object features like distance to nearest object.

Reference: AXIOM paper Section 2, Equations 6-7.
"""

import numpy as np


class RecurrentMixtureModel:
    """
    Mixture model over mixed continuous-discrete slot state tuples.

    Each component has a Gaussian likelihood over continuous features and
    Categorical likelihoods over each discrete feature.
    """

    def __init__(self, max_components: int, alpha_rmm: float = 1.0):
        self.max_components = max_components
        self.alpha_rmm = alpha_rmm
        self.num_active_components = 0
        self.component_params: list[dict] = []

    def infer_switch(
        self,
        continuous_features: np.ndarray,
        discrete_features: dict,
    ) -> np.ndarray:
        """
        Infer the rMM assignment variable and, through it, the tMM switch state.

        Args:
            continuous_features: slot position + interaction features (from C and g).
            discrete_features: dict with identity code, action, reward.

        Returns:
            Posterior over rMM components and implied tMM switch distribution.
        """
        raise NotImplementedError

    def update_params(
        self,
        continuous_features: np.ndarray,
        discrete_features: dict,
        assignment: int,
    ):
        """M-step: update Gaussian and Categorical parameters for the assigned component."""
        raise NotImplementedError

    def expand_if_needed(
        self, continuous_features: np.ndarray, discrete_features: dict
    ):
        """Grow a new rMM component if existing ones can't explain the data."""
        raise NotImplementedError
