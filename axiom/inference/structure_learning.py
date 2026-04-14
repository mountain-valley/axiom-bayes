"""
Online structure learning (growing heuristic) for AXIOM mixture models.

For each new datapoint, compares the posterior-predictive log-density under existing
components against a new-component threshold τ derived from the stick-breaking prior.
If no existing component exceeds τ and capacity remains, instantiates a new component.

Reference: AXIOM paper Section 2.1, "Fast structure learning".
"""

import numpy as np


def compute_threshold(prior_predictive_log_density: float, alpha: float) -> float:
    """
    Compute the new-component threshold τ = log p₀(y) + log α.

    Args:
        prior_predictive_log_density: log p₀(y) under an empty component.
        alpha: stick-breaking concentration parameter.
    """
    return prior_predictive_log_density + np.log(alpha)


def should_expand(
    component_scores: np.ndarray,
    threshold: float,
    num_active: int,
    max_components: int,
) -> bool:
    """Decide whether to create a new mixture component."""
    if num_active >= max_components:
        return False
    if len(component_scores) == 0:
        return True
    return float(np.max(component_scores)) < threshold


def assign_or_expand(
    data_point: np.ndarray,
    component_scores: np.ndarray,
    threshold: float,
    num_active: int,
    max_components: int,
) -> tuple[int, bool]:
    """
    Hard-assign data to best component, or signal that a new one is needed.

    Returns:
        (component_index, is_new): index of the assigned component and whether it's new.
    """
    if should_expand(component_scores, threshold, num_active, max_components):
        return num_active, True
    return int(np.argmax(component_scores)), False
