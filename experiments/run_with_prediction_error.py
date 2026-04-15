"""Run AXIOM while logging per-step next-state prediction error.

This script mirrors the official AXIOM training loop but writes one extra CSV
column: `Next-State Prediction Error`. It is intentionally optional so the
default workflow can continue to use upstream `vendor/axiom/main.py` unchanged.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AXIOM_DIR = PROJECT_ROOT / "vendor" / "axiom"
if str(AXIOM_DIR) not in sys.path:
    sys.path.insert(0, str(AXIOM_DIR))

import defaults  # type: ignore  # noqa: E402

import gameworld.envs  # noqa: E402,F401 (registers env IDs)
import gymnasium  # noqa: E402

from axiom import infer as ax  # noqa: E402


def _compute_next_state_prediction_error(
    prev_x_layer: jnp.ndarray,
    curr_x_layer: jnp.ndarray,
    switches: jnp.ndarray,
    tracked_obj_ids_layer: jnp.ndarray,
    transitions: jnp.ndarray,
    num_rmm_features: int,
) -> float:
    """Compute mean one-step MSE over tracked objects at a timestep."""
    state_prev = prev_x_layer[:, : prev_x_layer.shape[1] - num_rmm_features]
    state_curr = curr_x_layer[:, : curr_x_layer.shape[1] - num_rmm_features]

    selected = transitions[switches.astype(jnp.int32)]
    predicted = (selected[..., :-1] * state_prev[:, None, :]).sum(-1) + selected[..., -1]

    per_object_mse = ((state_curr - predicted) ** 2).mean(-1)
    tracked = tracked_obj_ids_layer.astype(bool)
    has_tracked = bool(np.asarray(tracked).any())
    if not has_tracked:
        return float("nan")

    tracked_mse = jnp.where(tracked, per_object_mse, jnp.nan)
    return float(np.asarray(jnp.nanmean(tracked_mse)))


def run(config) -> None:
    key = jr.PRNGKey(config.seed)
    np.random.seed(config.seed)

    if config.precision_type == "float64":
        jax.config.update("jax_enable_x64", True)

    env = gymnasium.make(
        f"Gameworld-{config.game}-v0",
        perturb=config.perturb,
        perturb_step=config.perturb_step,
    )

    rewards: list[float] = []
    expected_utility: list[float] = []
    expected_info_gain: list[float] = []
    num_components: list[float] = []
    next_state_prediction_error: list[float] = []

    obs, _ = env.reset()
    obs = obs.astype(np.uint8)
    reward = 0.0

    key, subkey = jr.split(key)
    carry = ax.init(subkey, config, obs, env.action_space.n)

    bmr_buffer = None, None

    for t in tqdm(range(config.num_steps)):
        key, subkey = jr.split(key)
        action, carry, plan_info = ax.plan_fn(subkey, carry, config, env.action_space.n)

        best = jnp.argsort(plan_info["rewards"][:, :, 0].sum(0))[-1]
        expected_utility.append(
            float(np.asarray(plan_info["expected_utility"][:, best, :].mean(-1).sum(0)))
        )
        expected_info_gain.append(
            float(np.asarray(plan_info["expected_info_gain"][:, best, :].mean(-1).sum(0)))
        )
        num_components.append(float(np.asarray(carry["rmm_model"].used_mask.sum())))

        obs, reward, done, truncated, info = env.step(action)
        obs = obs.astype(np.uint8)
        rewards.append(float(reward))

        update = True
        remap_color = False
        if (
            config.remap_color
            and config.perturb is not None
            and t + 1 >= config.perturb_step
            and t < config.perturb_step + 20
        ):
            update = False
            remap_color = True

        dyn_layer = config.layer_for_dynamics
        prev_x_layer = carry["x"][dyn_layer]

        carry, rec = ax.step_fn(
            carry,
            config,
            obs,
            jnp.array(reward),
            action,
            num_tracked=0,
            update=update,
            remap_color=remap_color,
        )

        pred_error = _compute_next_state_prediction_error(
            prev_x_layer=prev_x_layer,
            curr_x_layer=rec["x"][dyn_layer],
            switches=rec["switches"],
            tracked_obj_ids_layer=rec["tracked_obj_ids"][dyn_layer],
            transitions=carry["tmm_model"].transitions,
            num_rmm_features=config.rmm.num_features,
        )
        next_state_prediction_error.append(pred_error)

        if done:
            obs, _ = env.reset()
            obs = obs.astype(np.uint8)
            reward = 0
            carry, _ = ax.step_fn(
                carry,
                config,
                obs,
                jnp.array(reward),
                jnp.array(0),
                num_tracked=0,
                update=False,
            )

        if (t + 1) % config.prune_every == 0:
            key, subkey = jr.split(key)
            new_rmm, pairs, *bmr_buffer = ax.reduce_fn_rmm(
                subkey,
                carry["rmm_model"],
                *bmr_buffer,
                n_samples=config.bmr_samples,
                n_pairs=config.bmr_pairs,
            )
            carry["rmm_model"] = new_rmm

    out_csv = AXIOM_DIR / f"{config.game.lower()}.csv"
    with out_csv.open(mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Step",
                "Reward",
                "Average Reward",
                "Cumulative Reward",
                "Expected Utility",
                "Expected Info Gain",
                "Num Components",
                "Next-State Prediction Error",
            ]
        )
        for i in range(len(rewards)):
            writer.writerow(
                [
                    i,
                    rewards[i],
                    float(np.asarray(jnp.mean(jnp.array(rewards[max(0, i - 1000) : max(i, 1)])))),
                    float(np.asarray(sum(jnp.array(rewards[max(0, i - 1000) : i])))),
                    expected_utility[i],
                    expected_info_gain[i],
                    num_components[i],
                    next_state_prediction_error[i],
                ]
            )

    env.close()


def main() -> None:
    os.environ.setdefault("WANDB_MODE", "disabled")
    config = defaults.parse_args(sys.argv[1:])
    run(config)


if __name__ == "__main__":
    main()
