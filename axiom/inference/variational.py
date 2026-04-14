"""
Coordinate-ascent variational inference for AXIOM.

Implements the streaming E-step (forward filtering over latent states) and
M-step (natural parameter updates for all mixture model parameters).
Runs once per timestep for online learning.

Reference: AXIOM paper Section 2, Equations 8-9.
"""

import numpy as np

from axiom.models.smm import SlotMixtureModel
from axiom.models.imm import IdentityMixtureModel
from axiom.models.tmm import TransitionMixtureModel
from axiom.models.rmm import RecurrentMixtureModel


def e_step(
    observation: np.ndarray,
    smm: SlotMixtureModel,
    imm: IdentityMixtureModel,
    tmm: TransitionMixtureModel,
    rmm: RecurrentMixtureModel,
) -> dict:
    """
    Variational E-step: update posterior over latent states for one timestep.

    1. sMM assigns pixels to slots, yielding slot latents O_t.
    2. iMM infers identity codes from slot color/shape features.
    3. rMM infers switch states from multi-object features.
    4. tMM predicts next slot states given switch states.

    Returns:
        Dict of updated latent state posteriors.
    """
    raise NotImplementedError


def m_step(
    latent_states: dict,
    observation: np.ndarray,
    smm: SlotMixtureModel,
    imm: IdentityMixtureModel,
    tmm: TransitionMixtureModel,
    rmm: RecurrentMixtureModel,
):
    """
    Variational M-step: update parameters of all mixture models using
    sufficient statistics from the E-step and the current observation.
    """
    raise NotImplementedError
