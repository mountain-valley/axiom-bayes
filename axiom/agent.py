"""
Top-level AXIOM agent.

Wires together the four mixture models, variational inference, structure learning,
BMR, and the active inference planner into a single agent interface that can
interact with a Gameworld environment.
"""

import numpy as np

from axiom.models.smm import SlotMixtureModel
from axiom.models.imm import IdentityMixtureModel
from axiom.models.tmm import TransitionMixtureModel
from axiom.models.rmm import RecurrentMixtureModel
from axiom.planning.active_inference import Planner


class AXIOMAgent:
    """
    AXIOM agent: object-centric active inference with expanding mixture models.

    Maintains the full generative model (sMM + iMM + tMM + rMM), performs
    streaming variational inference, and selects actions via expected free energy.
    """

    def __init__(self, config: dict):
        self.config = config
        self.step_count = 0
        self.rng = np.random.default_rng(config.get("seed", 0))

        self.smm = SlotMixtureModel(
            max_slots=config.get("max_slots", 16),
            alpha_smm=config.get("alpha_smm", 1.0),
        )
        self.imm = IdentityMixtureModel(
            max_types=config.get("max_types", 10),
            alpha_imm=config.get("alpha_imm", 1.0),
        )
        self.tmm = TransitionMixtureModel(
            max_modes=config.get("max_modes", 32),
            state_dim=config.get("state_dim", 8),
            alpha_tmm=config.get("alpha_tmm", 1.0),
        )
        self.rmm = RecurrentMixtureModel(
            max_components=config.get("max_rmm_components", 256),
            alpha_rmm=config.get("alpha_rmm", 1.0),
        )
        self.planner = Planner(
            num_actions=config.get("num_actions", 5),
            horizon=config.get("planning_horizon", 8),
            num_rollouts=config.get("num_rollouts", 128),
        )

        self.bmr_interval = config.get("bmr_interval", 500)
        self.slot_latents: np.ndarray | None = None

    def observe(self, observation: np.ndarray, reward: float):
        """
        Process a new observation and reward. Runs one cycle of:
        1. sMM: parse pixels into slot latents
        2. iMM: infer identity codes
        3. rMM: infer switch states (with object interaction features)
        4. tMM: predict dynamics
        5. M-step: update all model parameters
        6. Structure learning: expand models if needed
        7. BMR: merge rMM components periodically
        """
        self.step_count += 1
        raise NotImplementedError

    def act(self) -> int:
        """Select an action using active inference planning."""
        if self.slot_latents is None:
            return self.rng.integers(self.planner.num_actions)
        return self.planner.select_action(self.slot_latents, self, self.rng)

    def should_run_bmr(self) -> bool:
        return self.step_count > 0 and self.step_count % self.bmr_interval == 0
