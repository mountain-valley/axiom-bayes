"""
Bayesian Model Reduction (BMR) for the rMM.

Every ΔT_BMR steps, samples rMM component pairs, scores their mutual expected
log-likelihoods, and greedily merges candidates that decrease the expected free
energy of the multinomial distributions over reward and tMM switch state.

Reference: AXIOM paper Section 2.1, "Bayesian Model Reduction (BMR)".
"""

import numpy as np


def select_merge_candidates(
    num_active_components: int,
    max_pairs: int = 2000,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    """Sample up to max_pairs pairs of active rMM components to evaluate for merging."""
    raise NotImplementedError


def score_merge(
    component_a: dict,
    component_b: dict,
    sampled_data: np.ndarray,
) -> float:
    """
    Score a candidate merge by computing the change in expected free energy.

    Returns negative value if merge improves (decreases) free energy.
    """
    raise NotImplementedError


def perform_bmr(
    rmm_params: list[dict],
    num_active: int,
    max_pairs: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[list[dict], int]:
    """
    Run one round of Bayesian Model Reduction on rMM components.

    Returns:
        Updated component params and new active count after merges.
    """
    raise NotImplementedError
