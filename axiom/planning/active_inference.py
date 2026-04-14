"""
Active inference planning for AXIOM.

Evaluates candidate action sequences (policies) by rolling out imagined
trajectories through the learned world model and scoring them with expected
free energy: a sum of expected utility (reward seeking) and information gain
(epistemic exploration via rMM Dirichlet counts).

Reference: AXIOM paper Section 2.2, Equation 10.
"""

import numpy as np


class Planner:
    """Expected free energy planner over discrete action sequences."""

    def __init__(
        self,
        num_actions: int,
        horizon: int = 8,
        num_rollouts: int = 128,
        utility_weight: float = 1.0,
        info_gain_weight: float = 1.0,
    ):
        self.num_actions = num_actions
        self.horizon = horizon
        self.num_rollouts = num_rollouts
        self.utility_weight = utility_weight
        self.info_gain_weight = info_gain_weight

    def select_action(
        self,
        slot_latents: np.ndarray,
        world_model: object,
        rng: np.random.Generator,
    ) -> int:
        """
        Select the best first action by evaluating sampled policies.

        1. Sample num_rollouts random action sequences of length horizon.
        2. For each, roll out imagined trajectories using the world model.
        3. Score each with expected free energy (utility + info gain).
        4. Return the first action of the best-scoring policy.
        """
        raise NotImplementedError

    def compute_expected_utility(
        self,
        imagined_rewards: np.ndarray,
    ) -> float:
        """E_q[log p(r | O, π)] accumulated over the planning horizon."""
        raise NotImplementedError

    def compute_information_gain(
        self,
        rmm_dirichlet_counts: np.ndarray,
        imagined_assignments: np.ndarray,
    ) -> float:
        """D_KL(q(α_rmm | O, π) || q(α_rmm)) — expected parameter information gain."""
        raise NotImplementedError
